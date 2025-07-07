export CUDA_VISIBLE_DEVICES=1
export MODEL_NAME='runwayml/stable-diffusion-v1-5'
export DATASET_DIR='/home/Data/maxiao/Polyp/TrainDatasetLoRA'
export OUTPUT_DIR="outputs/polyp_lora"
export RESOLUTION=512

torchrun diffusers/examples/text_to_image/train_text_to_image_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATASET_DIR \
    --output_dir=$OUTPUT_DIR \
    --resolution=$RESOLUTION \
    --train_batch_size=8 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing \
    --max_train_steps=10000 \
    --checkpointing_steps=5000 \
    --learning_rate=1e-04 --lr_scheduler='cosine' --lr_warmup_steps=0 \
    --mixed_precision='fp16'
