from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from config import TrainConfig

def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

def freeze_layer(model, num_layers: int):
    for parameter in model.parameters():
        parameter.requires_grad = True

    num_layers_to_freeze = TrainConfig.layers_to_freeze  
    for layer_index, layer in enumerate(model.model.encoder.layers):
        if layer_index < len(model.model.encoder.layers) - num_layers_to_freeze:
            for parameter in layer.parameters():
                parameter.requires_grad = False

    for layer_index, layer in enumerate(model.model.decoder.layers):
        if layer_index < len(model.model.encoder.layers) - num_layers_to_freeze:
            for parameter in layer.parameters():
                parameter.requires_grad = False