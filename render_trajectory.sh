export CUDA_VISIBLE_DEVICES=7 && \
python render_trajectory.py \
    -m output/hotdog \
    --num_frames 1000 \
    --output_video