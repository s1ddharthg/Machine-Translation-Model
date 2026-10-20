from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from model import load_model_and_tokenizer, freeze_layer
from metrics import build_compute_metrics
from preprocessor import preprocess_function, tokenize_datasets
import torch
from config import TrainConfig
from transformers.trainer_utils import get_last_checkpoint

def resolve_checkpoint(output_dir: str,resume_from_checkpoint: str | None,):
    if resume_from_checkpoint is None:
        return None

    if resume_from_checkpoint == "latest":
        checkpoint = get_last_checkpoint(output_dir)

        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {output_dir}"            )
        return checkpoint

    checkpoint_path = Path(resume_from_checkpoint)

    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}"        )

    if not (checkpoint_path / "trainer_state.json").exists():
        raise ValueError("The selected directory does not appear to be a Hugging Face Trainer checkpoint.")

    return str(checkpoint_path)

def run():

    raw_dataset = load_dataset(TrainConfig.dataset_name, revision=TrainConfig.dataset_revision)
    model, tokenizer = load_model_and_tokenizer(TrainConfig.model_name)
    freeze_layer(model, TrainConfig.layers_to_freeze)

    train_dataset, test_dataset = tokenize_datasets(raw_dataset, tokenizer)
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
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    checkpoint = resolve_checkpoint(str(TrainConfig.output_dir), TrainConfig.resume_from_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(TrainConfig.output_dir / "best_model"))
    tokenizer.save_pretrained(str(TrainConfig.output_dir / "best_model"))

if __name__ == "__main__":
    run()