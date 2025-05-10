import pandas as pd 
import numpy as np
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from datasets import load_dataset
dataset = load_dataset('csv', data_files='marketing_strategy_dataset_30.csv')

dataset = load_dataset('csv', data_files='marketing_strategy_dataset_30.csv')
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

model_checkpoint = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess(examples):
    inputs = examples['input_text']
    targets = examples['target_text']
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding='max_length')
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=256, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset.map(preprocess, batched=True)

# Load model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./bart-finetuned-marketing",
    eval_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True, 
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
)


trainer.train()

# Save the model
trainer.save_model("./bart-finetuned-marketing")
tokenizer.save_pretrained("./bart-finetuned-marketing")



tokenizer = AutoTokenizer.from_pretrained("./bart-finetuned-marketing")
model = AutoModelForSeq2SeqLM.from_pretrained("./bart-finetuned-marketing")

article = "Full article about the company..."

# Tokenize properly with explicit max_length
inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True, padding='max_length')

# Generate better summaries
summary_ids = model.generate(
    inputs['input_ids'],
    num_beams=4,
    length_penalty=2.0,
    max_length=256,
    min_length=50,
    early_stopping=True
)

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(output)
