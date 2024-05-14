import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def save_model(tokenizer, model, max_len, model_name):
    model_folder = "model"
    model_path = os.path.join(model_folder, model_name)

    os.makedirs(model_folder, exist_ok=True)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
    
    # Guardar max_len
    max_len_path = os.path.join(model_path, "max_len.json")
    with open(max_len_path, "w") as f:
        json.dump(max_len, f)