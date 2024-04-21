import sys
sys.path.append("./")
import numpy as np
import cv2
import os
from scene.dataset_readers import readCamerasFromTransforms
import json

eps = 1e-6

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
    H, W = img.size

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
        cv2.arrowedLine(img, point_pair[0], point_pair[1], (255, 255, 255), 4, tipLength=0.5)
    return img

def generate_handles_and_masks(source_path, points, r):
    points = np.array(points, dtype=float)
    N = points.shape[0]
    ones = np.ones((N, 2, 1), dtype=float)
    points_homo = np.concatenate([points, ones], axis=-1)
    
    mask_folder = os.path.join(source_path, "train_mask")
    handle_folder = os.path.join(source_path, "train_handle")
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(handle_folder, exist_ok=True)
    
    train_cam_infos = readCamerasFromTransforms(source_path, "transforms_train.json", white_background=True)
    
    with open(os.path.join(source_path, "transforms_train.json"), 'r') as f:
        json_content = json.load(f)
    
    for cam in train_cam_infos:

        local_points_homo = points_homo @ cam.w2c.T
        local_points = (local_points_homo / (local_points_homo[..., -1:] + eps))[..., :-1]
        image_points_homo = local_points @ cam.intrinsic.T
        image_points = (image_points_homo / (image_points_homo[..., -1:] + eps))[..., :-1]
        image_points = image_points.astype(int)
        
        img_with_points = draw_points(cv2.cvtColor(np.copy(cam.image), cv2.COLOR_BGR2RGB), image_points)
        mask = generate_mask(cam.image, image_points, r)
        mask = (mask.astype(int) * 255).reshape(*mask.shape, 1)
        
        alpha = 0.8
        cv2.imwrite(os.path.join(mask_folder, os.path.basename(cam.image_path)), mask)
        cv2.imwrite(os.path.join(handle_folder, os.path.basename(cam.image_path)), img_with_points * alpha + mask * (1-alpha))
        
        for i, view in enumerate(json_content["frames"]):
            if os.path.basename(view['file_path']) in os.path.basename(cam.image_path):
                json_content["frames"][i]['points'] = image_points.tolist()
    
    with open(os.path.join(source_path, "transforms_train.json"), 'w') as f:
        json.dump(json_content, f, indent=4)


if __name__ == '__main__':
    """
        Generate 2D handles and 2D masks for training views. 
        Save masks to source_path/train_mask/
        Save handle visualizations to source_path/train_handle/
        Add 2D handles to json file and save to source_path/transforms_train_with_handles.json
    """
    
    source_path = "data/nerf_synthetic/hotdog"
    
    # points in shape (N x 2 x 3), [[start_point, end_point], ...] 
    points = [
        [[0.43, -0.36, 0.05], [0.43, -0.75, 0.05]],
    ]
    
    r = 80 # The radius of the capsule shape on 2D
    
    generate_handles_and_masks(source_path, points, r)
    

        
    
    
    
    
    