from config import TrainConfig

def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["hi"] for ex in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=TrainConfig.max_length, truncation=True)
    labels = tokenizer(targets, max_length=TrainConfig.max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def tokenize_datasets(dataset):
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

    return tokenized_datasets_validation, tokenized_datasets_test