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

# DragDiffusion
## Generate drag points:
```
python scripts/generate_handles.py
```
