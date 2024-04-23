
"""
This script is to render a round-view video visualization of an object.
"""


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
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import GaussianModel
import json

from utils.camera_utils import *
import cv2

from scene.cameras import MiniCam
from utils.graphics_utils import *

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


def render_views(views:list, dataset:ModelParams, iteration:int, pipeline:PipelineParams, args):
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
        
        frames = []
        for idx, view in enumerate(tqdm(views, desc="Rendering progress", ncols=100)):
            rendering = render(view, gaussians, pipeline, background)
            img = rendering["render"]
            if not args.output_video:
                torchvision.utils.save_image(img, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            frame = (img.cpu()*255).clamp(0,255).numpy().transpose(1,2,0).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frames.append(frame)
        
        if args.output_video:
            video_out = cv2.VideoWriter(os.path.join(render_path, 'video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), fps=30, 
                                frameSize=(frames[0].shape[1], frames[0].shape[0]))
            for frame in tqdm(frames,ncols=100):
                video_out.write(frame)
            cv2.destroyAllWindows()
            video_out.release()



if __name__ == '__main__':

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
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
    render_views(cams, model.extract(args), args.iteration, pipeline.extract(args), args)