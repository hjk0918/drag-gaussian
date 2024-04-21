export CUDA_VISIBLE_DEVICES=0

python drag.py \
    -s data/nerf_synthetic/hotdog \
    -m output/hotdog \
    --start_ply_path output/hotdog/point_cloud/iteration_30000/point_cloud.ply \
    ---num_drag_steps 50 \
    --iterations 5000 \
    --eval