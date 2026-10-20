from config import TrainConfig
from datasets import load_dataset

def load_translation_dataset(dataset_name: str, revision: str | None = None,):
    kwargs = {}

    if revision is not None:
        kwargs["revision"] = revision

    dataset = load_dataset(dataset_name, **kwargs)

    required_splits = {"train", "validation", "test"}
    missing = required_splits - set(dataset.keys())

    if missing:
        raise ValueError(f"Missing dataset splits: {sorted(missing)}")

    return dataset

def preprocess_function(examples, tokenizer):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["hi"] for ex in examples["translation"]]

    model_inputs = tokenizer(inputs, max_length=TrainConfig.max_length, truncation=True)
    labels = tokenizer(targets, max_length=TrainConfig.max_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

def tokenize_datasets(dataset, tokenizer):
    tokenized_datasets_validation = dataset['validation'].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["validation"].column_names,
        batch_size=2,
        fn_kwargs={"tokenizer": tokenizer},
    )

    tokenized_datasets_test = dataset['test'].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["test"].column_names,
        batch_size=2,
        fn_kwargs={"tokenizer": tokenizer},
    )

    return tokenized_datasets_validation, tokenized_datasets_test