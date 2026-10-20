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