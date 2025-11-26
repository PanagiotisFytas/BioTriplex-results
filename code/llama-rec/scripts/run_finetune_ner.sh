#export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
#export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128'

python recipes/quickstart/finetuning/finetuning.py \
    --use_peft \
    --peft_method lora \
    --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
    --output_dir 'models/3.1-8B_biotripv2_ner' \
    --batch_size_training 1 \
    --batching_strategy "padding" \
    --weight_decay 0.2 \
    --num_epochs 10 \
    --dataset biotriplex_ner_dataset \
    --context_length 10000 \
    --use_entity_tokens_as_targets False \
    --entity_special_tokens False \
    --use_fast_kernels True \
    --bidirectional_attention_in_entity_tokens False \
    --use_wandb True

# infer on test
python recipes/quickstart/inference/local_inference/inference.py \
    --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
    --peft_model 'models/3.1-8B_biotripv2_ner' \
    --max_new_tokens 2000 \
    --top_p 1 \
    --top_k 200 \
    --repetion_penalty 1.0 \
    --temperature 0.6 \
    --share_gradio True \
    --enable_salesforce_content_safety False \
    --full_dataset \
    --ner_dataset True \
    --use_entity_tokens_as_targets False \
    --entity_special_tokens False \
    --shift_entity_tokens False \
    --bidirectional_attention_in_entity_tokens False \
    --dataset_mode 'test' \
    --prefix 'models/'
