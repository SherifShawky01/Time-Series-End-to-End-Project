# app_cnn_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import pandas as pd

# Define CNN model (same as before)
class CNN1D(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 16, 3, padding=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features * 16, 1)
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return self.fc(x)

# Load trained model
input_dim = 14
model = CNN1D(input_dim)
model.load_state_dict(torch.load("cnn_model.pt", map_location=torch.device("cpu")))
model.eval()

# FastAPI app
app = FastAPI()


class InputData(BaseModel):
    features: list  # a list of 13 floats

@app.get("/")
def read_root():
    return {"message": "CNN Regressor for Time Series Forecasting"}

@app.post("/predict")
def predict(data: InputData):
    try:
        x = torch.tensor(data.features, dtype=torch.float32).unsqueeze(0)  # shape (1, 13)
        with torch.no_grad():
            pred = model(x).item()
        return {"prediction": pred}
    except Exception as e:
        return {"error": str(e)}