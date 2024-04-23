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
import json
import torch
import numpy as np
import shutil
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
from arguments import ModelParams, PipelineParams, OptimizationParams, DragParams
from utils.camera_utils import render_samples, visualize_drag
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.drag_utils import DragWrapper

"""
DragGaussian training pipeline:

Preparation: The initial gaussian scene should be trained and saved as ply using the 3DGS official train.py
Repeat:
    1. Render training views.
    2. Apply one step of diffusion update on rendered training view images.
    3. Replace the training dataset with the updated training view images.
    4. Train the gaussians until converge with the new dataset. 
"""
# Selected dragging viewpoints for hotdog scene
selected_viewpoint_indices = [
    0, 1, 2, 3, 4, 5, 7, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 
    21, 22, 23, 24, 26, 28, 29, 30, 31, 32, 33, 37, 38, 
    41, 42, 43, 44, 45, 47, 48, 51, 52, 53, 55, 56, 57, 58, 60,
    61, 62, 63, 64, 67, 68, 71, 73, 74, 76, 78, 80, 
    83, 85, 86, 88, 90, 92, 96, 98, 99
]
selected_viewpoint_indices = selected_viewpoint_indices[::6]


def drag(model_args, opt_args, pipe_args, drag_args):
    # Init gaussians
    gaussians = GaussianModel(model_args.sh_degree)
    scene = Scene(model_args, gaussians, shuffle=False)
    gaussians.training_setup(opt_args)
    viewpoint_cams = scene.getTrainCameras().copy()
    viewpoint_cams = [viewpoint_cams[i] for i in selected_viewpoint_indices]
    
    bg_color = [1, 1, 1] if model_args.white_background else [0, 0, 0]
    bg = torch.rand((3), device="cuda") if opt_args.random_background \
        else torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Init DragWrapper    
    images = np.stack([(cam.original_image.cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8) for cam in viewpoint_cams], axis=0)
    masks = np.stack([cam.mask for cam in viewpoint_cams], axis=0)
    points = [cam.points.astype(np.float32) for cam in viewpoint_cams]
    
    full_h, full_w = images.shape[1:3]
    drag_args.sup_res_h = int(full_h)
    drag_args.sup_res_w = int(full_w)
    drag_wrapper = DragWrapper(images, points, masks, drag_args)
    
    global_gs_iter = 0
    render_dir = os.path.join(model_args.model_path, "renders")
    if os.path.exists(render_dir):
        shutil.rmtree(render_dir)

    os.makedirs(render_dir, exist_ok=True)

    # Iteratively update the dataset and train gaussians.
    # for n_iters steps
    #   1. run dragdiffusion from n_drag_views viewpoints for n_pix_step steps using rendered images
    #   2. train gaussian from n_gs_views viewpoints (n_gs_views iters) using edited images
    for iter in range(1, drag_args.n_iters + 1):
        selected_views = np.random.permutation(len(viewpoint_cams))[:drag_args.n_drag_views]

        # Render training views.
        rendered_training_imgs = []
        with torch.no_grad():
            for view in selected_views:
                viewpoint_cam = viewpoint_cams[view]
                image = torch.clamp(render(viewpoint_cam, gaussians, pipe_args, bg)['render'], 0.0, 1.0)
                rendered_training_imgs.append(image)
        rendered_training_imgs = torch.stack(rendered_training_imgs, dim=0).half().to(drag_args.drag_device)
        
        # Apply one step diffusion update on rendered training view images.
        updated_training_imgs = drag_wrapper.update(rendered_training_imgs, selected_views)
        
        # Replace the training dataset with the updated training view images.
        for i, ind in enumerate(selected_views):
            viewpoint_cams[ind].original_image = updated_training_imgs[i].squeeze().float().to(drag_args.gs_device)
        
        # Fine-tune gasussians with the new dataset. 
        train_gaussians(gaussians, viewpoint_cams, scene, bg, model_args, opt_args, pipe_args, drag_args)

        # Render training views and save.
        if iter % drag_args.vis_interval == 0:
            # render_samples(gaussians, viewpoint_cams, render, bg, pipe_args, render_dir, iter)
            visualize_drag(selected_views, images, masks, viewpoint_cams, points, drag_wrapper.handle_points, 
                           iter, render_dir)
        
        # Save gaussians
        if iter % drag_args.gs_save_interval == 0:
            scene.drag_gaussian_save(iter)

    render_samples(gaussians, viewpoint_cams, render, bg, pipe_args, render_dir, drag_args.n_iters)
    visualize_drag(selected_views, images, masks, viewpoint_cams, points, drag_wrapper.handle_points,
                   drag_args.n_iters, render_dir)
    scene.drag_gaussian_save(drag_args.n_iters)
    

def train_gaussians(gaussians, viewpoint_cams, scene, bg, model_args, opt_args, pipe_args, drag_args):
    """ Fine-tune the gaussians. """
    
    viewpoint_idx_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(drag_args.n_gs_views), desc="Fine-tuning Gaussians")
    for iteration in range(1, drag_args.n_gs_views + 1):

        # gaussians.update_learning_rate(iteration)
        
        if not viewpoint_idx_stack:
            viewpoint_idx_stack = list(range(len(viewpoint_cams)))
        viewpoint_cam = viewpoint_cams[viewpoint_idx_stack.pop(randint(0, len(viewpoint_idx_stack)-1))]
        
        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe_args, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt_args.lambda_dssim) * Ll1 + opt_args.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == drag_args.n_gs_views:
                progress_bar.close()
            
            # Densification
            # if iteration < opt_args.densify_until_iter:
            #     # Keep track of max radii in image-space for pruning
            #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            #     if iteration > opt_args.densify_from_iter and iteration % opt_args.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt_args.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt_args.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
            #     if iteration % opt_args.opacity_reset_interval == 0 or (model_args.white_background and iteration == opt_args.densify_from_iter):
            #         gaussians.reset_opacity()

            # Optimizer step
            if iteration < drag_args.n_gs_views:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Dragging script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    dp = DragParams(parser)

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
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    drag(lp.extract(args), op.extract(args), pp.extract(args), dp.extract(args))

    # All done
    print("\nTraining complete.")
