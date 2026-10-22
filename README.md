# English → Hindi Machine Translation

<p align="center">
  <strong>A Transformer-powered English to Hindi translation system.</strong>
  <br />
  Fine-tune a pretrained MarianMT model, evaluate translation quality with SacreBLEU, and serve predictions through a lightweight Flask API.
</p>

<p align="center">
  <a href="https://github.com/s1ddharthg/Machine-Translation-Model/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/s1ddharthg/Machine-Translation-Model?style=for-the-badge&color=2EA44F" alt="License" />
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Architecture-Transformer-8B5CF6?style=flat-square" alt="Transformer Architecture" />
  <img src="https://img.shields.io/badge/Model-MarianMT-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="MarianMT" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/API-Flask-000000?style=flat-square&logo=flask&logoColor=white" alt="Flask" />
</p>

---

## Overview

**English → Hindi Machine Translation** is an experimental neural machine translation project that converts English text into Hindi using a pretrained sequence-to-sequence Transformer model.

Built with Hugging Face Transformers and PyTorch, the project fine-tunes the [Helsinki-NLP/opus-mt-en-hi](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi) model on the [IIT Bombay English–Hindi dataset](https://huggingface.co/datasets/cfilt/iitb-english-hindi). The resulting model can be used to translate English sentences into Hindi text through a simple HTTP API.

The project covers the complete machine learning workflow, from dataset preprocessing and model fine-tuning to evaluation and inference.

### How it works

```text
       English Input
             |
             v
      Text Preprocessing
             |
             v
   MarianMT Transformer Model
             |
             v
      Hindi Translation
             |
             v
       JSON API Response
```

### Features

- **Transformer-based translation:** Uses a pretrained MarianMT encoder-decoder model for English-to-Hindi translation.
- **Fine-tuning pipeline:** Trains the pretrained model on parallel English–Hindi sentence pairs.
- **Dataset preprocessing:** Loads and tokenizes bilingual text using Hugging Face Datasets and tokenizers.
- **Translation evaluation:** Uses SacreBLEU to measure translation quality.
- **Configurable training:** Keeps model and training parameters in a centralized configuration module.
- **Checkpoint support:** Stores training checkpoints for recovery and continued training, subject to the configured training setup.
- **REST API:** Exposes a Flask endpoint that accepts English text and returns its Hindi translation.
- **Local inference:** Loads the trained model from disk for translation without requiring an external translation API.

## Tech Stack

| Technology | Purpose |
|---|---|
| [Python](https://www.python.org/) | Core programming language |
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | Transformer model loading and fine-tuning |
| [MarianMT](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi) | Pretrained English-to-Hindi translation model |
| [Hugging Face Datasets](https://huggingface.co/docs/datasets) | Dataset loading and preprocessing |
| [SacreBLEU](https://github.com/mjpost/sacrebleu) | Translation evaluation |
| [Flask](https://flask.palletsprojects.com/) | HTTP API and inference endpoint |

## Model and Dataset

### Base model

[**Helsinki-NLP/opus-mt-en-hi**](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi)

The project uses MarianMT, an encoder-decoder Transformer architecture designed for neural machine translation. The pretrained model provides an initial translation capability that can be adapted to the project's training data through fine-tuning.

### Training dataset

[**cfilt/iitb-english-hindi**](https://huggingface.co/datasets/cfilt/iitb-english-hindi)

The IIT Bombay English–Hindi corpus contains parallel sentences in English and Hindi. These sentence pairs provide the source and target examples used to train the translation model.

### Evaluation

**SacreBLEU** is used to evaluate generated Hindi translations against reference translations. It provides a standardized BLEU score for comparing translation output.

> Evaluation scores depend on the dataset split, preprocessing, training configuration, and model checkpoint. Actual scores should be reported only after running the evaluation pipeline.

## Project Structure

```text
Machine-Translation-Model/
├── app.py                      # Flask translation API
├── requirements.txt            # Python dependencies
├── src/
│   ├── __init__.py
│   ├── config.py               # Model and training configuration
│   ├── metrics.py              # SacreBLEU evaluation
│   ├── model.py                # Model loading and layer freezing
│   ├── preprocessor.py          # Dataset loading and tokenization
│   └── train.py                # Training entry point
├── finetuned-nlp-en-hi/
│   ├── checkpoint-*/            # Training checkpoints
│   └── best_model/              # Saved model for inference
├── .gitignore
├── LICENSE
└── README.md
```

*The generated checkpoint and model directories appear after training and are not tracked by Git.*

## Getting Started

### Prerequisites

- Python 3.10 or newer
- Git
- Internet access for downloading the pretrained model, tokenizer, dataset, and metric dependencies
- Sufficient disk space for the dataset, model weights, and checkpoints
- A compatible CPU or CUDA-enabled GPU

GPU acceleration is recommended for faster training, but actual requirements depend on model settings, batch size, and available memory.

### 1. Clone the repository

```bash
git clone https://github.com/s1ddharthg/Machine-Translation-Model.git
cd Machine-Translation-Model
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

First, install a PyTorch version compatible with your hardware using the [official installation selector](https://pytorch.org/get-started/locally/).

Then install the project's dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify that PyTorch is installed correctly:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

A `True` CUDA result indicates that PyTorch can access a compatible CUDA device.

## Training

Run the training pipeline from the repository root:

```bash
python -m src.train
```

The pipeline loads the configured pretrained model and dataset, preprocesses the bilingual examples, fine-tunes the model, and evaluates translation quality using the configured metrics.

### Configuration

Training parameters are defined in `TrainConfig` in `src/config.py`. Model-related settings are defined in `ModelConfig`.

Configure these settings before training:

- **Model name:** The pretrained English-to-Hindi MarianMT checkpoint.
- **Dataset:** The English–Hindi parallel corpus.
- **Training arguments:** Epochs, batch size, learning rate, and other supported Trainer parameters.
- **Device and precision:** CPU or GPU execution and compatible numerical precision.
- **Checkpoint directory:** Where intermediate model states are saved.
- **Model output directory:** Where the final model is saved.
- **Evaluation settings:** Metric configuration and evaluation frequency.

The expected inference directory is:

```text
finetuned-nlp-en-hi/best_model/
```

Ensure that the training script actually saves the model and tokenizer to this location before starting the API.

### Resuming training

If checkpoint resumption is enabled in the training configuration, a previous run can be continued from a compatible saved checkpoint rather than starting from scratch.

For example, the Hugging Face Trainer supports:

```python
trainer.train(resume_from_checkpoint=True)
```

This resumes from the latest eligible checkpoint in the configured output directory. To resume from a specific checkpoint, pass its directory path instead.

Checkpoint resumption requires a valid checkpoint created by the Trainer. Saving only the final model weights is not equivalent to preserving the full training state.

## Run the Translation API

After training and saving the model, start the Flask application:

```bash
python app.py
```

The local development server should be available at:

```text
http://127.0.0.1:5000
```

The API loads the configured model from disk and uses it to translate incoming English text into Hindi.

### Endpoint

`POST /translate`

**Request headers**

```http
Content-Type: application/json
```

**Request body**

```json
{
  "text": "Hello, how are you?"
}
```

**Example request using cURL**

```bash
curl -X POST http://127.0.0.1:5000/translate \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello, how are you?"}'
```

**Successful response**

```json
{
  "translation": "हैलो, तुम कैसे हो?"
}
```

The actual Hindi output depends on the trained checkpoint and its generation settings.

### Input validation

The endpoint expects a non-empty string in the `text` field.

| Scenario | Expected behavior |
|---|---|
| Valid English text | Returns a Hindi translation |
| Missing `text` field | HTTP `400` |
| Empty or whitespace-only text | HTTP `400` |
| Invalid JSON or wrong field type | Should be rejected with HTTP `400` if handled by the application |

The exact response format for error cases depends on the validation implemented in `app.py`.

## Example Translation

**Input — English**

```text
Technology is changing the way we communicate.
```

**Output — Hindi**

```text
प्रौद्योगिकी हमारे संवाद करने के तरीके को बदल रही है।
```

*Illustrative example only. The actual output is generated by the model and may differ.*

## Implementation Notes

### Sequence-to-sequence learning

The model learns to generate a target Hindi sequence conditioned on an input English sequence. During training, tokenized English sentences serve as model inputs, while the corresponding Hindi sentences provide the target labels.

### Tokenization

The tokenizer converts text into token IDs and prepares the sequences for the Transformer. Input and target sequences are padded or truncated according to the configured preprocessing strategy.

### Fine-tuning

Fine-tuning adapts the pretrained translation model to the project's training corpus. Training settings, checkpointing behavior, and layer-freezing choices are controlled by the project configuration and model-loading implementation.

### Inference

At inference time, the Flask application tokenizes the English input, runs the model's generation method, and decodes the generated token sequence into Hindi text.

## Troubleshooting

**Model directory not found**

Ensure training has completed and that `ModelConfig.model_dir` points to the directory containing the saved model and tokenizer.

**CUDA out-of-memory error**

Reduce the per-device batch size, shorten maximum sequence lengths, or use gradient accumulation. Use mixed precision only when it is supported by the selected hardware and training configuration.

**Training starts from scratch unexpectedly**

Check that checkpoint saving is enabled and that the resume path points to a valid Trainer checkpoint.

**`revision` argument errors**

The Transformers `revision` argument identifies a model or