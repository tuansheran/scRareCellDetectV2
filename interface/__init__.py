
import os
import torch
from interface.graphSAGE import GraphSAGE

MODEL_DIR = os.path.dirname(__file__)  
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "entire_model1.pth")

def get_model(model_name="model_1"):
    model_paths = {
        "model_1": os.path.join(MODEL_DIR, "entire_model1.pth"),
        "model_2": os.path.join(MODEL_DIR, "entire_model3.pth"),
        "model_3": os.path.join(MODEL_DIR, "entire_model4.pth"),
    }


    model_path = model_paths.get(model_name)
    
    if model_path is None:
        raise ValueError(f"Model '{model_name}' not found at path: {model_path}")

    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    
    return model