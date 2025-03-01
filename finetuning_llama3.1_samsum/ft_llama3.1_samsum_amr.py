import transformers
import torch
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sentence_transformers
import nltk
import glob
import datasets
import peft

cache_directory = '/scratch/alpine/anra7539' # For caching model init
checkpoint_directory = "/scratch/alpine/anra7539/llama3.1_qlora_finetuned_samsum_amr" # For fine-tuned model checkpoint
inference_results = '/projects/anra7539/projects/representation_efficacy/samsum_amr_summarization_qllama3.1_finetuned/generated_summaries.json'

samsum_train = pd.read_csv("/projects/anra7539/projects/representation_efficacy/samsum_train.csv")
samsum_val = pd.read_csv("/projects/anra7539/projects/representation_efficacy/samsum_test.csv")
samsum_test = pd.read_csv("/projects/anra7539/projects/representation_efficacy/SamSum_val.csv")

samsum_train.dropna(subset = ['amr'], inplace = True)
samsum_val.dropna(subset = ['amr'], inplace = True)
samsum_test.dropna(subset = ['amr'], inplace = True)

samsum_train['only_amr'] = samsum_train.amr.map(lambda x: "\n".join(x.split("\n")[1:]))
samsum_val['only_amr'] = samsum_val.amr.map(lambda x: "\n".join(x.split("\n")[1:]))
samsum_test['only_amr'] = samsum_test.amr.map(lambda x: "\n".join(x.split("\n")[1:]))

samsum_train.dropna(inplace = True)
samsum_val.dropna(inplace = True)

train_df = datasets.Dataset.from_pandas(samsum_train)
val_df = datasets.Dataset.from_pandas(samsum_val)

name = "meta-llama/Llama-3.1-8B-Instruct"
device = "cuda"

device_map = "auto"
model = transformers.AutoModelForCausalLM.from_pretrained(name,
                                                          load_in_8bit=True,
                                                          trust_remote_code=True,
                                                          device_map=device_map,
                                                          cache_dir=cache_directory)

tokenizer = transformers.AutoTokenizer.from_pretrained(name, truncation_side="left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


def tokenize_function(example):
    start_prompt = '''Summarize the following conversation between two people in 1-2 sentences.
Ensure the summary captures the core intent, actions, and resolution.
Avoid examples, quotes, or minor details.
You may also use the provided Abstract Meaning Representation (AMR) structure of the conversation to your aid.\n\n### Conversation:\n'''
    end_prompt = '\n\n### Summary: '
    
    full_texts = [f'''{start_prompt}{dialogue}\n\n### Conversation AMR:\n{amr}{end_prompt}{summary}''' for dialogue,amr,summary in zip(example['dialogue'], example['only_amr'], example['summary'])]
    
    encoding = tokenizer(
        full_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=4096
    )
    
    input_ids = encoding.input_ids
    labels = input_ids.clone()

    example["input_ids"] = input_ids.to(device)
    example["labels"] = labels.to(device)
    return example

batch_size = 2

tokenized_datasets_train = train_df.map(tokenize_function, batched=True, batch_size = batch_size)
tokenized_datasets_val = val_df.map(tokenize_function, batched=True, batch_size = batch_size)

tokenized_datasets_train = tokenized_datasets_train.remove_columns(['only_amr', 'amr', 'dialogue', 'summary', '__index_level_0__'])
tokenized_datasets_val = tokenized_datasets_val.remove_columns(['only_amr', 'amr', 'dialogue', 'summary'])

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_model_params = sum(p.numel() for p in model.parameters())
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

lora_config = peft.LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=peft.TaskType.CAUSAL_LM
)

peft_model = peft.get_peft_model(model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

peft_training_args = transformers.TrainingArguments(
    output_dir=checkpoint_directory,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=1e-3,
    num_train_epochs=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
    greater_is_better=False,
    save_total_limit=1,
    ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
)


peft_trainer = transformers.Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets_train,
    eval_dataset=tokenized_datasets_val,
    callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=3)],
)

peft_trainer.train()

peft_model_path=checkpoint_directory

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

# Inference

base_model = transformers.AutoModelForCausalLM.from_pretrained(name,
                                                          load_in_8bit = True,
                                                          trust_remote_code = True,
                                              device_map={"": torch.cuda.current_device()},
                                             cache_dir=cache_directory)
tokenizer = transformers.AutoTokenizer.from_pretrained(name, truncation_side = "left")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

peft_model = peft.PeftModel.from_pretrained(base_model, 
                                       checkpoint_directory,
                                       is_trainable=False)

def summarization(dialogue, amr, prompt):
    with torch.no_grad():
        input_text = f'''{prompt}\n\n### Conversation:\n{dialogue}\n\n### Conversation AMR:\n{amr}\n\n### Summary:'''
        input_tokens = tokenizer(input_text, 
                                 return_tensors = "pt", 
                                 truncation = True, 
                                 max_length = 4096).to(device)
        
        outputs = model.generate(**input_tokens, 
                                 max_new_tokens=64, 
                                 temperature=0.1, 
                                 top_p=0.9, 
                                 repetition_penalty=1.1, 
                                 pad_token_id=tokenizer.eos_token_id, do_sample=True)        
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summary = full_response.split("### Summary:")[1].strip().split("\n")[0].strip()
    return summary

prompt = f'''Summarize the following conversation in 1-2 sentences.
Ensure the summary captures the core intent, actions, and resolution.
Avoid examples, quotes, or minor details.
You may also use the provided Abstract Meaning Representation (AMR) structure of the conversation to your aid.'''

output_file = inference_results

if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        try:
            existing_data = [json.loads(line) for line in f]
        except json.JSONDecodeError:
            existing_data = []
else:
    existing_data = []

processed_indices = {item['index'] for item in existing_data}

with open(output_file, 'a') as f:
    for i in tqdm(range(len(samsum_test))):
        if i in processed_indices:
            continue 

        summary = summarization(samsum_test.dialogue[i], 
                                samsum_test.only_amr[i], 
                                prompt, 
                                model, 
                                tokenizer)
        
        result = {
            "index": i,
            "dialogue": samsum_test.dialogue[i],
            "summary": samsum_test.summary[i],
            "prediction": summary
        }

        f.write(json.dumps(result) + "\n")
        f.flush()