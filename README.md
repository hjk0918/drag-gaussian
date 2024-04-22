# DragGaussian
Here we hardcode to use the hotdog scene in the following scripts.
## 1. Train 3DGS
Train 3DGS of the original scene and save as ply file.  
```bash
python train.py \
    -s data/nerf_synthetic/hotdog \
    -m output/hotdog
```

## 2. Generate drag points and masks
```
python scripts/generate_handles.py
```
## 3. Train DragGaussian
```bash
python drag.py \
    -s data/nerf_synthetic/hotdog \
    -m output/hotdog \
    --start_ply_path output/hotdog/point_cloud/iteration_30000/point_cloud.ply \
    ---num_drag_steps 50 \
    --iterations 5000 \
    --eval \
    --output_path output/hotdog
```

# 3DGS
## Installation
```
conda env create --file environment.yml
conda activate drag_gaussian
```

## Dataset
Donwload nerf_synthetic dataset from [this link](https://drive.google.com/file/d/18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG/view?usp=share_link) and unzip it to `data/nerf_synthetic/` directory.

## Training
```bash
python train.py \
    -s data/nerf_synthetic/hotdog \
    -m output/hotdog
```

## Render Round-view Video
```bash
python render_trajectory.py \
    -m output/hotdog \
    --num_frames 1000 \
    --output_video
```
