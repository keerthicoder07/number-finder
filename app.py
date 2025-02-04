from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import uvicorn
import base64
from PIL import Image
import io

# Load the trained CNN model
model = load_model("mnist_cnn.h5")

app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    image_data = base64.b64decode(data["data"].split(",")[1])  # Extract base64 part
    image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  
    image = np.array(image)

    # Preprocess Image
    image = image.astype("float32") / 255.0  # Normalize
    image = np.reshape(image, (-1, 28, 28, 1))  # Reshape for model input

    # Predict Digit
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    return {"predicted_digit": int(predicted_digit)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
