"""Research demo only: FastAPI service around the Protocol A custom CNN (notebooks/02, cell 6).

NOT a medical device and not clinically validated. Requires: pip install fastapi uvicorn python-multipart pillow.
Set MODEL_PATH to a trained 128x128 grey-scale CNN (e.g. outputs/cnn_50/tb_cnn_model.h5)."""
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO
from PIL import Image

# #####Initialize FastAPI app
app = FastAPI(title="TB Classification API", description="Upload a chest X-ray image to detect tuberculosis.", version="1.0")

# Load the trained CNN model
MODEL_PATH = "tb_cnn_model.h5"
model = load_model(MODEL_PATH)

# Define image size for CNN model
IMG_SIZE = 128

# Function to preprocess the image
def preprocess_image(image_data):
    img = Image.open(BytesIO(image_data)).convert("L") 
    img = img.resize((IMG_SIZE, IMG_SIZE))  
    img_array = np.array(img) / 255.0 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = np.expand_dims(img_array, axis=-1)  
    return img_array

# #####Home route
@app.get("/")
def home():
    return {"message": "Welcome to the TB Classification API!"}

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file and preprocess it
        image_data = await file.read()
        img_array = preprocess_image(image_data)

        # Make prediction
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        # Determine class label
        result = "Tuberculosis Detected" if confidence > 0.5 else "Normal"

        return JSONResponse(content={"diagnosis": result, "confidence": confidence})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run API with Uvicorn
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8006)
