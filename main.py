from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from model import load_model_and_tokenizer, freeze_layer
from metrics import build_compute_metrics
from preprocessor import preprocess_function, tokenize_datasets
import torch
from config import TrainConfig

def run():

    raw_dataset = load_dataset(TrainConfig.dataset_name)
    model, tokenizer = load_model_and_tokenizer(TrainConfig.model_name)
    freeze_layer(model, TrainConfig.num_layers_to_freeze)

    validation_dataset, test_dataset = tokenize_datasets(raw_dataset, tokenizer)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        train_dataset=test_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()
    trainer.save_model(str(TrainConfig.output_dir / "best_model"))
    tokenizer.save_pretrained(str(TrainConfig.output_dir / "best_model"))

if __name__ == "__main__":
    run()