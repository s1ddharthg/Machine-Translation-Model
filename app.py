from pathlib import path
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import ModelConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not path.exists(ModelConfig.model_dir):
    raise FileNotFoundError(f"Model directory does not exist: {ModelConfig.model_dir}")

tokenizer = AutoTokenizer.from_pretrained(ModelConfig.model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(ModelConfig.model_dir)
model.eval()

app = Flask(__name__)

@app.post("/translate")
def translate():
    body = request.get_json(silent=True)
    text = body.get("text") if isinstance(body, dict) else None

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text' must be a non-empty string"}), 400

    inputs = tokenizer(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return jsonify({"translation": translation})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)