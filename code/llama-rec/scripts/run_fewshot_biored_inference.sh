#!/bin/bash

python recipes/quickstart/inference/local_inference/inference.py \
    --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
    --max_new_tokens 50 \
    --top_p 1 \
    --top_k 10 \
    --repetion_penalty 1.0 \
    --temperature 0.2 \
    --share_gradio True \
    --enable_salesforce_content_safety False \
    --full_dataset \
    --biored_qakshot_dataset True \
    --return_neg_relations False \
    --dataset_mode 'test' \
    --num_of_shots 0 \
    --prefix 'models/biored_qa_0'



python recipes/quickstart/inference/local_inference/inference.py \
    --model_name 'meta-llama/Llama-3.1-8B-Instruct' \
    --max_new_tokens 50 \
    --top_p 1 \
    --top_k 10 \
    --repetion_penalty 1.0 \
    --temperature 0.2 \
    --share_gradio True \
    --enable_salesforce_content_safety False \
    --full_dataset \
    --biored_qakshot_dataset True \
    --return_neg_relations False \
    --dataset_mode 'test' \
    --num_of_shots 5 \
    --prefix '/home/pf376/rds/hpc-work/llama-models/bio_infer/biored_qa_5'



