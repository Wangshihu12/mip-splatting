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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size: float, scaling_modifier = 1.0, override_color = None, subpixel_offset=None):
    """
    渲染3D高斯点云场景，生成2D图像。
    
    该函数是3D高斯点云渲染的核心函数，负责将3D高斯点投影到2D屏幕空间，
    并执行光栅化过程生成最终的渲染图像。
    
    Args:
        viewpoint_camera: 视点相机对象，包含相机参数和变换矩阵
        pc: 高斯模型对象，包含所有3D高斯点的参数
        pipe: 渲染管道配置，包含各种渲染选项
        bg_color: 背景颜色张量，必须在GPU上
        kernel_size: 高斯核大小，控制渲染质量
        scaling_modifier: 缩放修饰符，默认为1.0
        override_color: 覆盖颜色，如果提供则使用此颜色替代计算的颜色
        subpixel_offset: 子像素偏移，用于光线抖动
    
    Returns:
        dict: 包含渲染结果的字典
            - render: 渲染的图像
            - viewspace_points: 视空间中的点坐标
            - visibility_filter: 可见性过滤器
            - radii: 高斯点在屏幕上的半径
    """
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # 创建零张量，用于获取2D屏幕空间均值的梯度
    # 这个张量会被光栅化器更新为屏幕空间坐标
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        # 尝试保留梯度，用于反向传播
        screenspace_points.retain_grad()
    except:
        # 如果保留梯度失败，继续执行
        pass

    # 设置光栅化配置
    # 计算视场角的正切值，用于透视投影
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)  # 水平视场角的正切值
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)  # 垂直视场角的正切值

    # 如果没有提供子像素偏移，创建零偏移
    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")
        
    # 创建光栅化设置对象，包含所有渲染参数
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),  # 图像高度
        image_width=int(viewpoint_camera.image_width),    # 图像宽度
        tanfovx=tanfovx,                                  # 水平视场角正切值
        tanfovy=tanfovy,                                  # 垂直视场角正切值
        kernel_size=kernel_size,                          # 高斯核大小
        subpixel_offset=subpixel_offset,                  # 子像素偏移
        bg=bg_color,                                      # 背景颜色
        scale_modifier=scaling_modifier,                  # 缩放修饰符
        viewmatrix=viewpoint_camera.world_view_transform, # 世界到视图的变换矩阵
        projmatrix=viewpoint_camera.full_proj_transform,  # 投影变换矩阵
        sh_degree=pc.active_sh_degree,                    # 当前激活的球谐函数度数
        campos=viewpoint_camera.camera_center,            # 相机中心位置
        prefiltered=False,                                # 是否预过滤
        debug=pipe.debug                                  # 调试模式
    )

    # 创建光栅化器实例
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取3D高斯点的参数
    means3D = pc.get_xyz                    # 3D位置
    means2D = screenspace_points            # 2D屏幕空间位置（将被光栅化器更新）
    opacity = pc.get_opacity_with_3D_filter # 透明度（应用了3D滤波器）

    # 协方差矩阵处理
    # 如果提供了预计算的3D协方差矩阵，使用它；否则由光栅化器从缩放/旋转计算
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # 在Python中预计算3D协方差矩阵
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        # 使用缩放和旋转参数，让光栅化器计算协方差
        scales = pc.get_scaling_with_3D_filter  # 缩放参数（应用了3D滤波器）
        rotations = pc.get_rotation             # 旋转参数

    # 颜色处理
    # 如果提供了预计算的颜色，使用它们；否则根据需要预计算球谐函数到RGB的转换
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            # 在Python中将球谐函数转换为RGB颜色
            # 重塑特征张量以匹配球谐函数格式
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            # 计算从相机中心到每个点的方向向量
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            # 归一化方向向量
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # 评估球谐函数得到RGB颜色
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            # 将颜色值限制在[0, 1]范围内
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            # 直接使用球谐函数系数，让光栅化器处理转换
            shs = pc.get_features
    else:
        # 使用覆盖颜色
        colors_precomp = override_color

    # 光栅化可见的高斯点到图像，获取它们在屏幕上的半径
    rendered_image, radii = rasterizer(
        means3D = means3D,           # 3D位置
        means2D = means2D,           # 2D屏幕空间位置
        shs = shs,                   # 球谐函数系数
        colors_precomp = colors_precomp,  # 预计算的颜色
        opacities = opacity,         # 透明度
        scales = scales,             # 缩放参数
        rotations = rotations,       # 旋转参数
        cov3D_precomp = cov3D_precomp)  # 预计算的3D协方差矩阵

    # 那些被视锥体剔除或半径为0的高斯点不可见
    # 它们将被排除在用于分割标准的数值更新之外
    return {"render": rendered_image,           # 渲染的图像
            "viewspace_points": screenspace_points,  # 视空间中的点坐标
            "visibility_filter" : radii > 0,    # 可见性过滤器（半径大于0的点）
            "radii": radii}                     # 高斯点在屏幕上的半径
