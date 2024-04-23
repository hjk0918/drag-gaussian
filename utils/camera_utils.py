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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import os
import shutil as sh
import cv2
import PIL.Image as Image
import torch
from torchvision.utils import save_image

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  points=cam_info.points, mask=cam_info.mask)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def render_samples(gaussian_model, viewpoint_cams, render_func, bg, pipe_args, output_folder, drag_step):
    print(f"Rendering samples of drag step {drag_step}... ", end="")
    # render_folder = os.path.join(output_folder, f'drag_step_{drag_step}/renders')
    # if os.path.exists(render_folder):
    #     sh.rmtree(render_folder)
    # os.makedirs(render_folder, exist_ok=True)
    
    for viewpoint_idx, viewpoint_cam in enumerate(viewpoint_cams):    
        image = torch.clamp(
            render_func(
                viewpoint_cam,
                gaussian_model,
                pipe_args,
                bg 
            )["render"], 0.0, 1.0
        )

        image_gt_rendered = torch.concatenate((viewpoint_cam.original_image.cuda(), image), dim=2)
        save_image(image_gt_rendered, os.path.join(output_folder, f'gs_{drag_step:03d}_{viewpoint_idx:02d}.png'))
        
    print("Done.")

def draw_points(img, points):
    # points in shape (N x 2 x 3)
    for idx, point_pair in enumerate(points):
        cv2.circle(img, tuple(point_pair[0]), 5, (255, 0, 0), -1)
        cv2.circle(img, tuple(point_pair[1]), 5, (0, 0, 255), -1)
        cv2.arrowedLine(img, tuple(point_pair[0]), tuple(point_pair[1]), (255, 255, 255), 2, tipLength=0.1)
    return img

def visualize_drag(img_inds, original_imgs, masks, viewpoint_cams, points, cur_handles,
                   drag_step, output_folder):
    for ind in img_inds:
        img_orig = original_imgs[ind]
        mask = masks[ind]
        img_new = viewpoint_cams[ind].original_image.cpu().squeeze().permute(1, 2, 0)
        img_new = np.ascontiguousarray(img_new.numpy().astype(np.float32))

        img_points = points[ind]
        img_handles = cur_handles[ind]

        img_orig = img_orig.astype(np.float32) / 255
        img_orig[~mask] *= 0.5

        init_pairs = []
        cur_pairs = []
        for point, handle in zip(img_points, img_handles):
            # point in (x, y), handle in (y, x)
            init_handle = (point[0, 0], point[0, 1])
            target = (point[1, 0], point[1, 1])
            handle = handle.to("cpu", torch.int32).numpy()
            cur_handle = (handle[1], handle[0])

            init_pairs.append((init_handle, target))
            cur_pairs.append((cur_handle, target))

        img_orig = draw_points(img_orig, init_pairs)
        img_new = draw_points(img_new, cur_pairs)

        img = np.concatenate((img_orig, img_new), axis=1)
        img = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(output_folder, f'drag_{drag_step:03d}_{ind:02d}.png'))
