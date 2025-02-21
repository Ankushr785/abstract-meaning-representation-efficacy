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

from sklearn.model_selection import train_test_split


cache_directory = '/scratch/alpine/anra7539'
checkpoint_directory = "/scratch/alpine/anra7539/llama3.1_qlora_finetuned_squad_amr"
inference_results = '/projects/anra7539/projects/representation_efficacy/squad_ft_amr_answers_qllama3.1/predicted_answers.json'

squad = datasets.load_dataset("squad_v2", cache_dir=cache_directory)
squad_train = squad['train'].to_pandas()
squad_test = squad['validation'].to_pandas()

squad_train['answer_list'] = squad_train.answers.map(lambda x: x['text'])
squad_test['answer_list'] = squad_test.answers.map(lambda x: x['text'])

squad_test_amrs = pd.read_csv("/projects/anra7539/projects/representation_efficacy/squad_val_amrs.csv")
squad_train_amrs = pd.read_csv("/projects/anra7539/projects/representation_efficacy/squad_train_amrs.csv")

squad_test_amrs.drop_duplicates(inplace=True)
squad_test_amrs.reset_index(drop=True, inplace=True)
squad_test_amrs['only_amr'] = squad_test_amrs.amr.map(lambda x: "\n".join(x.split("\n")[1:]))

squad_train_amrs.drop_duplicates(inplace=True)
squad_train_amrs.reset_index(drop=True, inplace=True)
squad_train_amrs['only_amr'] = squad_train_amrs.amr.map(lambda x: "\n".join(x.split("\n")[1:]))

squad_test = squad_test.merge(squad_test_amrs, on=['context'], how='right').reset_index(drop=True)
squad_train = squad_train.merge(squad_train_amrs, on=['context'], how='right').reset_index(drop=True)

squad_training, squad_val = train_test_split(squad_train, test_size=0.25, random_state=2025)

for df in [squad_training, squad_val, squad_test]:
    df['final_answer'] = df.answer_list.map(lambda x: x[0] if len(x) > 0 else 'unanswerable')

train_df = datasets.Dataset.from_pandas(squad_training)
val_df = datasets.Dataset.from_pandas(squad_val)

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

# model.gradient_checkpointing_enable()

def tokenize_function(example):
    start_prompt = '''Answer the given question based on the context.
    If the question can't be answered based on the information in the context, return "unanswerable".
    You will not return anything except the answer.
    You may also use the provided linearized Abstract Meaning Representation (AMR) structure of the paragraph to aid in reasoning.'''
    end_prompt = '\nAnswer: '
    prompt = [f'''{start_prompt}\n\nContext: {context}\n\nAMR: {amr}\n\nQuestion: {question}{end_prompt}''' 
              for context,question,amr in zip(example['context'], example['question'], example['only_amr'])]
    example['input_ids'] = tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=4096).input_ids.to(device)
    example['labels'] = tokenizer(example["final_answer"], padding="max_length", truncation=True, return_tensors="pt", max_length=4096).input_ids.to(device)
    return example

batch_size = 2

tokenized_datasets_train = train_df.map(tokenize_function, batched=True, batch_size=batch_size)
tokenized_datasets_val = val_df.map(tokenize_function, batched=True, batch_size=batch_size)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_model_params = sum(p.numel() for p in model.parameters())
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

lora_config = peft.LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
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
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
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

def qna(question, context, amr, prompt, model, tokenizer):
    with torch.no_grad():
        input_text = f'''{prompt}\n\nContext:{context}\n\nAMR:{amr}\n\nQuestion:{question}\nAnswer:'''
        input_tokens = tokenizer(input_text, 
                                 return_tensors = "pt", 
                                 truncation = True, 
                                 max_length = 4096).to(device)
    
        outputs = model.generate(**input_tokens, 
                                 max_new_tokens = 30, 
                                 pad_token_id = tokenizer.eos_token_id)
    
        answer = tokenizer.decode(outputs[0], skip_special_tokens = True).split("Answer:")[1].split("\n")[0].split(".")[0].strip()
    return answer

prompt = f'''Answer the given question based on the context.
If the question can't be answered based on the information in the context, return "unanswerable".
You will not return anything except the answer.
You may also use the provided linearized Abstract Meaning Representation (AMR) structure of the paragraph to aid in reasoning.'''

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
    for i in tqdm(range(len(squad_test))):
        if i in processed_indices:
            continue 
        
        answer = qna(squad_test.question[i], squad_test.context[i], squad_test.only_amr[i], prompt, peft_model, tokenizer)
        
        result = {
            "index": i,
            "context": squad_test.context[i],
            "question": squad_test.question[i],
            "answer_list": list(squad_test.answer_list[i]),
            "prediction": answer
        }

        f.write(json.dumps(result) + "\n")
        f.flush()