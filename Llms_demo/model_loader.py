import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name):
    model_folder = "model"
    model_path = os.path.join(model_folder, model_name)

    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
        # Cargar max_len
        max_len_path = os.path.join(model_path, "max_len.json")
        with open(max_len_path, "r") as f:
            max_len = json.load(f)

        return tokenizer, model, max_len
    else:
        print("Model not found locally.")
        return None, None, None