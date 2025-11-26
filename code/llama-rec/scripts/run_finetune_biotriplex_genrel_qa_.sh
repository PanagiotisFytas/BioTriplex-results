#!/bin/bash


python recipes/quickstart/finetuning/finetuning.py \
    --use_peft \
    --peft_method lora \
    --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
    --output_dir 'models/3.1-8B_biotriplex_genrel_qamc' \
    --batch_size_training 1 \
    --batching_strategy "padding" \
    --num_epochs 6 \
    --dataset biotriplex_qakshot_dataset \
    --num_of_shots 0 \
    --context_length 10000 \
    --use_entity_tokens_as_targets False \
    --entity_special_tokens False \
    --use_fast_kernels True \
    --upweight_minority_class False \
    --bidirectional_attention_in_entity_tokens False \
    --enable_fsdp False \
    --return_neg_relations False \
    --use_wandb True \
    --general_relations True


python recipes/quickstart/inference/local_inference/inference.py \
    --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
    --peft_model 'models/3.1-8B_biotriplex_genrel2_qamc' \
    --max_new_tokens 50 \
    --top_p 1 \
    --top_k 50 \
    --repetion_penalty 2.0 \
    --temperature 0.6 \
    --share_gradio True \
    --enable_salesforce_content_safety False \
    --full_dataset \
    --qa_kshot_dataset True \
    --num_of_shots 0 \
    --use_entity_tokens_as_targets False \
    --entity_special_tokens False \
    --shift_entity_tokens False \
    --bidirectional_attention_in_entity_tokens False \
    --return_neg_relations False \
    --dataset_mode 'test' \
    --general_relations True \
    --prefix 'models/'

