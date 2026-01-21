CUDA_VISIBLE_DEVICES=1 \
python /raid/danielchen/Mars-LS-challenge/Multimodal-Martian-Landslides-Dataset/implementation/generate_submission.py \
    --checkpoint /raid/danielchen/Mars-LS-challenge/Multimodal-Martian-Landslides-Dataset/implementation/checkpoints/run_20260121_202617/best_model.pth \
    --output_zip submission.zip \
    --patch_size 4