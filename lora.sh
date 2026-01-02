python finetune_script.py \
    --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
    --dataset_name="KhangTruong/NWPU_Split" \
    --resolution=256 --center_crop --random_flip \
    --train_batch_size=8 \
    --num_train_epochs=100 \
    --learning_rate=1e-05 \
    --max_grad_norm=1 \
    --revision="flax"\
    --output_dir="./sd-model-finetuned" \
    --caption_column="raw"

