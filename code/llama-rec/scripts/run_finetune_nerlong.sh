
# upsample minority class
python recipes/quickstart/finetuning/finetuning.py \
    --use_peft \
    --peft_method lora \
    --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
    --output_dir 'models/3.1-8B_biotrip_nerlong_upsample' \
    --batch_size_training 1 \
    --batching_strategy "padding" \
    --weight_decay 0.2 \
    --num_epochs 10 \
    --dataset biotriplex_nerlong_dataset \
    --context_length 10000 \
    --use_entity_tokens_as_targets False \
    --entity_special_tokens False \
    --use_fast_kernels True \
    --weighted_sampling True \
    --bidirectional_attention_in_entity_tokens False \
    --use_wandb True


# infer on test
python recipes/quickstart/inference/local_inference/inference.py \
    --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
    --peft_model 'models/3.1-8B_biotrip_nerlong_upsample' \
    --max_new_tokens 10000 \
    --top_p 1 \
    --top_k 50 \
    --repetion_penalty 1.0 \
    --temperature 0.6 \
    --share_gradio True \
    --enable_salesforce_content_safety False \
    --full_dataset \
    --nerlong_dataset True \
    --use_entity_tokens_as_targets False \
    --entity_special_tokens False \
    --shift_entity_tokens False \
    --bidirectional_attention_in_entity_tokens False \
    --dataset_mode 'test' \
    --prefix 'models/'

