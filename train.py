#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import numpy as np
import open3d as o3d
import cv2
import torch
import random
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    id_coords[..., 0] /= (width - 1)
    id_coords[..., 1] /= (height - 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    """
    3D高斯点云模型的训练函数，负责整个训练流程的管理和执行。
    
    Args:
        dataset: 数据集对象，包含训练和测试数据
        opt: 训练选项配置，包含学习率、迭代次数等超参数
        pipe: 渲染管道配置
        testing_iterations: 测试迭代次数列表
        saving_iterations: 保存模型迭代次数列表
        checkpoint_iterations: 检查点保存迭代次数列表
        checkpoint: 检查点文件路径，用于恢复训练
        debug_from: 调试起始迭代次数
    
    Returns:
        无返回值
    """
    # 初始化训练起始迭代次数
    first_iter = 0
    # 准备输出目录和日志记录器
    tb_writer = prepare_output_and_logger(dataset)
    # 创建3D高斯模型实例
    gaussians = GaussianModel(dataset.sh_degree)
    # 创建场景对象，管理数据集和高斯模型
    scene = Scene(dataset, gaussians)
    # 设置训练参数（优化器、学习率等）
    gaussians.training_setup(opt)
    
    # 如果提供了检查点文件，则恢复训练状态
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 设置背景颜色：白色背景或黑色背景
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 创建CUDA事件用于计时
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 获取训练和测试相机列表
    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras
    
    # 高分辨率相机索引列表
    # 筛选出图像宽度大于等于800像素的相机
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    # 计算3D滤波器，用于后续的密度控制
    gaussians.compute_3D_filter(cameras=trainCameras)

    # 初始化视角栈和损失记录变量
    viewpoint_stack = None
    ema_loss_for_log = 0.0  # 指数移动平均损失，用于日志记录
    # 创建进度条
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # 主训练循环
    for iteration in range(first_iter, opt.iterations + 1):        
        # 网络GUI连接管理
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                # 接收GUI发送的参数
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    # 渲染自定义相机视角的图像
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    # 将图像转换为字节格式发送给GUI
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # 记录迭代开始时间
        iter_start.record()

        # 更新学习率（根据迭代次数调整）
        gaussians.update_learning_rate(iteration)

        # 每1000次迭代增加球谐函数度数，最大到设定值
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择训练相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # 30%概率选择高分辨率相机进行训练
        # 这有助于提高模型在高分辨率图像上的表现
        if random.random() < 0.3 and dataset.sample_more_highres:
            viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
            
        # 渲染设置
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # 子像素偏移处理
        # TODO: 忽略边界像素
        if dataset.ray_jitter:
            # 生成随机子像素偏移，范围在[-0.5, 0.5]之间
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None
            
        # 执行渲染，获取图像、视空间点、可见性过滤器和半径信息
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, subpixel_offset=subpixel_offset)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 损失计算
        gt_image = viewpoint_cam.original_image.cuda()  # 获取真实图像
        # 如果启用重采样，对真实图像应用子像素偏移
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)

        # 计算L1损失
        Ll1 = l1_loss(image, gt_image)
        # 计算总损失：L1损失和SSIM损失的加权组合
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # 反向传播
        loss.backward()

        # 记录迭代结束时间
        iter_end.record()

        # 在不需要梯度计算的上下文中执行以下操作
        with torch.no_grad():
            # 进度条更新
            # 使用指数移动平均计算损失，使显示更平滑
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录训练报告和保存模型
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 密度控制阶段
            if iteration < opt.densify_until_iter:
                # 跟踪图像空间中高斯点的最大半径，用于后续剪枝
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 添加密度统计信息
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 在指定迭代范围内执行密度化和剪枝
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # 设置大小阈值：在透明度重置间隔后使用20，否则为None
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # 执行密度化和剪枝操作
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    # 重新计算3D滤波器
                    gaussians.compute_3D_filter(cameras=trainCameras)

                # 定期重置透明度
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 密度控制阶段结束后，每100次迭代更新3D滤波器
            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # 在训练结束前100次迭代不更新，避免影响最终结果
                    gaussians.compute_3D_filter(cameras=trainCameras)
        
            # 优化器步骤
            if iteration < opt.iterations:
                gaussians.optimizer.step()  # 执行参数更新
                gaussians.optimizer.zero_grad(set_to_none = True)  # 清零梯度

            # 保存检查点
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
