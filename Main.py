from fastapi import FastAPI
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

model = load_model("model.h5")
scaler = joblib.load("scaler.pkl")

@app.get("/")
def home():
    return {"message": "Churn Model Running"}

@app.post("/predict")
def predict(data: dict):
    input_data = np.array(data["features"]).reshape(1, -1)

    # scale input
    input_scaled = scaler.transform(input_data)

    # prediction
    pred = model.predict(input_scaled)
    result = int(pred[0][0] > 0.5)

    return {"prediction": result}
