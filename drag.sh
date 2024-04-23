python drag.py \
    -s data/nerf_synthetic/hotdog \
    -m output/hotdog \
    --start_ply_path output/hotdog/point_cloud/iteration_30000/point_cloud.ply \
    --prompt "a plate with two hotdogs" \
    --n_drag_views 100 \
    --n_gs_views 5000 \
    --n_iters 1 \
    --n_pix_step 80 \
    --eval \
    --output_path output/hotdog \
    --vis_interval 1