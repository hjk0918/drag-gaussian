export CUDA_VISIBLE_DEVICES=0,1

sh scripts/generate_baseline.py \
    -m output/chair_baseline \
    --output_path "output/chair_baseline" \
    --num_frames 30 \
    --output_video \
    --n_pix_step 80 \
    --white_background