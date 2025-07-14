---
base_model: /root/autodl-tmp/Llama3-8B-Chinese-Chat
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: train_2024-08-10-15-01-41
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_2024-08-10-15-01-41

This model is a fine-tuned version of [/root/autodl-tmp/Llama3-8B-Chinese-Chat](https://huggingface.co//root/autodl-tmp/Llama3-8B-Chinese-Chat) on the llama3_fine dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 6
- total_train_batch_size: 24
- total_eval_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 2.0

### Training results



### Framework versions

- PEFT 0.11.1
- Transformers 4.42.4
- Pytorch 2.3.0+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1