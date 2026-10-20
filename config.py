from dataclasses import dataclass
from pathlib import Path
MAX_LENGTH=256


@dataclass
class TrainConfig:
    model_name: str = "Helsinki-NLP/opus-mt-en-hi"
    dataset_name: str = "cfilt/iitb-english-hindi"
    output_dir: Path = Path("./finetuned-nlp-en-hi")
    max_length: int = 256
    train_batch_size: int = 8
    eval_batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 2000
    eval_steps: int = 250
    save_steps: int = 250
    logging_steps: int = 25
    layers_to_freeze: int = 10

    #Revison control
    model_revision: str = "main"
    dataset_revision: str | None = None
    resume_from_checkpoint: str | None = None
    save_total_limit: int = 3