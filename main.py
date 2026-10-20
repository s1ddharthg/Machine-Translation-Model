from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

dataset = load_dataset("cfilt/iitb-english-hindi")

maxLength = 256

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-hi")

def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["hi"] for ex in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=maxLength, truncation=True)
    labels = tokenizer(targets, max_length=maxLength, truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

tokenized_datasets_validation = dataset['validation'].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["validation"].column_names,
    batch_size=2
)

tokenized_datasets_test = dataset['test'].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["test"].column_names,
    batch_size=2)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

for parameter in model.parameters():
    parameter.requires_grad = True

num_layers_to_freeze = 10  
for layer_index, layer in enumerate(model.model.encoder.layers):
    if layer_index < len(model.model.encoder.layers) - num_layers_to_freeze:
        for parameter in layer.parameters():
            parameter.requires_grad = False

num_layers_to_freeze = 10  
for layer_index, layer in enumerate(model.model.decoder.layers):
    if layer_index < len(model.model.encoder.layers) - num_layers_to_freeze:
        for parameter in layer.parameters():
            parameter.requires_grad = False

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import evaluate

metric = evaluate.load("sacrebleu")

import numpy as np


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

model.to(device)
training_args = Seq2SeqTrainingArguments(
    f"finetuned-nlp-en-hi",
    gradient_checkpointing=True,
    per_device_train_batch_size=32,
    learning_rate=1e-5,
    warmup_steps=2,
    max_steps=2000,
    fp16=True,
    optim='adafactor',
    per_device_eval_batch_size=16,
    metric_for_best_model="eval_bleu",
    predict_with_generate=True,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets_test,
    eval_dataset=tokenized_datasets_validation,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()