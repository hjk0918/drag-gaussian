
"""
This script is to render a round-view video visualization of an object.
"""

import sys
sys.path.append("./")

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from os.path import join
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, DragParams
from scene import GaussianModel
import json

from utils.camera_utils import *
import cv2

from scene.cameras import MiniCam
from utils.graphics_utils import *
from utils.drag_utils import DragWrapper

eps = 1e-6

def poses2cams(poses, fx=900, fy=900, width=800, height=800):

    fovx = focal2fov(fx, width)
    fovy = focal2fov(fy, height)

    z_near = 0.01
    z_far = 100

    projection_matrix = getProjectionMatrix(z_near, z_far, fovx, fovy).transpose(0,1).float().float().cuda()

    cams = []
    for c2w in poses:
        c2w = np.vstack((c2w, np.array([0,0,0,1])))
        world_view_transform = torch.tensor(np.linalg.inv(c2w)).transpose(0, 1).float().cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        cam = MiniCam(width, height, fovy, fovx, z_near, z_far, 
                      world_view_transform, full_proj_transform)
        cams.append(cam)

    return cams

def point2seg_dist(x, y, x1, y1, x2, y2):
    dists = np.zeros_like(x, dtype=float)
    
    cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)
    sol1_mask = cross <= 0
    sol1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    dists[sol1_mask] = sol1[sol1_mask]
    
    d2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    sol2_mask = cross >= d2
    sol2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
    dists[sol2_mask] = sol2[sol2_mask]
    
    r = cross / d2
    px = float(x2 - x1) * r + x1
    py = float(y2 - y1) * r + y1
    sol3_mask = (~sol1_mask) * (~sol2_mask)
    sol3 = np.sqrt((x - px) ** 2 + (y - py) ** 2)
    dists[sol3_mask] = sol3[sol3_mask]

    return dists

def generate_mask(img, points, r):
    _, H, W = img.shape

    x = np.arange(H)
    y = np.arange(W)
    xv, yv = np.meshgrid(x, y)
    xv, yv = xv.astype(float), yv.astype(float)
    mask = np.zeros((H, W), dtype=bool)
    
    for idx, point_pair in enumerate(points):
        p1, p2 = point_pair[0], point_pair[1]
        x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
        cur_mask = point2seg_dist(xv, yv, x1, y1, x2, y2) <= r
        mask = np.logical_or(cur_mask, mask)
    
    return mask

def draw_points(img, points):
    # points in shape (N x 2 x 3)
    for idx, point_pair in enumerate(points):
        cv2.circle(img, tuple(point_pair[0]), 10, (255, 0, 0), -1)
        cv2.circle(img, tuple(point_pair[1]), 10, (0, 0, 255), -1)
        cv2.arrowedLine(img, tuple(point_pair[0]), tuple(point_pair[1]), (0, 255, 0), 2, tipLength=0.5)
    return img

def generate_points_and_masks(rendered_imgs, views, points):
    points = np.array(points, dtype=float)
    N = points.shape[0]
    ones = np.ones((N, 2, 1), dtype=float)
    points_homo = np.concatenate([points, ones], axis=-1)
    
    points_2d = []
    masks = []
    imgs_with_points = []
    for view, rendered_img in zip(views, rendered_imgs):
        
        w2c = np.array(view.world_view_transform.cpu())
        intrinsic = view.intrinsic

        local_points_homo = points_homo @ w2c
        local_points = (local_points_homo / (local_points_homo[..., -1:] + eps))[..., :-1]
        image_points_homo = local_points @ intrinsic.T
        image_points = (image_points_homo / (image_points_homo[..., -1:] + eps))[..., :-1]
        image_points = image_points.astype(int)
        
        mask = generate_mask(rendered_img, image_points, r=100)
        img_with_mask = rendered_img.cpu().numpy() * 0.8 + mask.reshape(1, *mask.shape) * 0.2
        img_with_mask = (img_with_mask*255).clip(0,255).transpose(1,2,0).astype(np.uint8)
        img_with_mask = cv2.cvtColor(img_with_mask, cv2.COLOR_RGB2BGR)
        img_with_points = draw_points(np.copy(img_with_mask), image_points)
        
        points_2d.append(image_points.astype(float))
        imgs_with_points.append(img_with_points)
        masks.append(mask)
    
    return points_2d, masks, imgs_with_points


def render_views(views:list, dataset:ModelParams, iteration:int, pipeline:PipelineParams, args, drag_args):
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        gaussians.load_ply(os.path.join(args.model_path, args.pcd))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if args.output_video:
            render_path = os.path.join(args.model_path, "trajectory_renders")
        else:
            render_path = os.path.join(args.model_path, "trajectory_renders/images")
        
        makedirs(render_path, exist_ok=True)
        
        rendered_imgs = []
        frames = []
        for idx, view in enumerate(tqdm(views, desc="Rendering progress", ncols=100)):
            rendering = render(view, gaussians, pipeline, background)
            img = torch.clamp(rendering["render"], 0.0, 1.0)
            rendered_imgs.append(img)
            
            frame = (img.cpu()*255).clamp(0,255).numpy().transpose(1,2,0).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)
        rendered_imgs = torch.stack(rendered_imgs, dim=0).half().to(drag_args.drag_device)
        frames = np.stack(frames, axis=0)
        
        points_3d = [
            # [[0.43, -0.36, 0.05], [0.43, -0.75, 0.05]], # hotdog
            [[-0.04, 0.58, 0.78], [-0.04, 0.7, 1.13]] # chair
        ]
        
        points, masks, imgs_with_points = generate_points_and_masks(rendered_imgs, views, points_3d)
        masks = np.stack(masks, axis=0)
        full_h, full_w = frames.shape[1:3]
        drag_args.sup_res_h = int(full_h)
        drag_args.sup_res_w = int(full_w)
    
    
    drag_wrapper = DragWrapper(frames, points, masks, drag_args)
    updated_imgs = drag_wrapper.update(rendered_imgs * 2.0 - 1.0, list(range(len(rendered_imgs))))
    updated_imgs = [cv2.cvtColor((x.squeeze(0).cpu()*255).clamp(0,255).numpy().transpose(1,2,0).astype(np.uint8), cv2.COLOR_RGB2BGR) for x in updated_imgs]
    
    video_out = cv2.VideoWriter(os.path.join(render_path, 'video_original.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), fps=1, 
                            frameSize=(frames[0].shape[1], frames[0].shape[0]))
    for frame in tqdm(frames,ncols=100):
        video_out.write(frame)
    cv2.destroyAllWindows()
    video_out.release()

    video_out = cv2.VideoWriter(os.path.join(render_path, 'video_handles.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), fps=1, 
                        frameSize=(frames[0].shape[1], frames[0].shape[0]))
    for frame in tqdm(imgs_with_points,ncols=100):
        video_out.write(frame)
    cv2.destroyAllWindows()
    video_out.release()
    
    video_out = cv2.VideoWriter(os.path.join(render_path, 'video_edited.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), fps=1, 
                            frameSize=(frames[0].shape[1], frames[0].shape[0]))
    for frame in tqdm(updated_imgs,ncols=100):
        video_out.write(frame)
    cv2.destroyAllWindows()
    video_out.release()



if __name__ == '__main__':

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    drag_params = DragParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_video", action="store_true", default=False, 
                        help="if store true, save the frames as a video, otherwise, save frames as separate images.")
    parser.add_argument("--num_frames", type=int, default=3600, help="number of frames to render")
    parser.add_argument("--pcd", type=str, default="point_cloud/iteration_7000/point_cloud.ply",
                        help="path to the model to render")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    args.global_illumination = False

    # Initialize system state (RNG)
    safe_state(args.quiet)

    world_up = np.array([0,0,1]).reshape(1,3)

    trajectory_poses = get_camera_trajectory(num_frames=args.num_frames, world_up=world_up)
    cams = poses2cams(trajectory_poses)
    render_views(cams, model.extract(args), args.iteration, pipeline.extract(args), args, drag_params.extract(args))