from fastapi import FastAPI, HTTPException, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle 
import numpy as np
import sklearn 

from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tempfile
import os

# Define the FastAPI app
app = FastAPI()

# Load the ML model using pickle
with open("lg.pkl", "rb") as f:
    model_d1 = pickle.load(f)


with open("db.pkl", "rb") as f:
    model_d2 = pickle.load(f)

model = load_model('skin_cancer_detector_80%.h5')


@app.get("/")
def optans():
    return 'HealthCare API By Optans Team'

@app.post("/skin_cancer")
async def create_upload_file(file: UploadFile = File(...)):
    # Read image using cv2
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        print("Error: Unable to load image.")
    else:
        # Resize the image
        width = 256 # New width
        height = 256  # New height
        resized_image = cv2.resize(img, (width, height))

        # Normalize the image by dividing by 255
        normalized_image = resized_image / 255.0

        # Convert RGB to BGR
        normalized_image_reshaped = np.expand_dims(normalized_image, axis=0)

        # Predict using the model
        result = model.predict(normalized_image_reshaped)

        # Existing dictionary mapping class indices to class names
        existing_dict = {
            0: 'actinic keratoses & ic',
            1: 'Basal cell',
            2: 'Benign keratosis',
            3: 'Dermatofibroma',
            4: 'Melanoma',
            5: 'Melanocytic nevi',
            6: 'Vascular skin'
        }

        # Extracting class probabilities
        class_probabilities = [
            {'class_name': existing_dict[i], 'probability': float(value)}
            for i, value in enumerate(result[0])
        ]

        # Sorting by probability
        class_probabilities_sorted = sorted(class_probabilities, key=lambda x: x['probability'], reverse=True)

        # Return classification result
        return class_probabilities_sorted



@app.post("/breast_cancer")
async def predict(
    radius_mean: float, perimeter_mean: float, area_mean: float,
    compactness_mean: float, concavity_mean: float, concave_points_mean: float,
    radius_se: float, perimeter_se: float, area_se: float,
    radius_worst: float, perimeter_worst: float, area_worst: float,
    compactness_worst: float, concavity_worst: float, concave_points_worst: float
):
    # Convert input data to numpy array
    input_data = np.array([[
        radius_mean, perimeter_mean, area_mean,
        compactness_mean, concavity_mean, concave_points_mean,
        radius_se, perimeter_se, area_se,
        radius_worst, perimeter_worst, area_worst,
        compactness_worst, concavity_worst, concave_points_worst
    ]])

    # Make predictions
    prediction = model_d1.predict(input_data)

    # Convert prediction to human-readable message
    if prediction[0] == 1:
        result = "The tumor is malignant (cancerous)."
    else:
        result = "The tumor is benign (non-cancerous)."

    return {"result": result , "pred_value":int(prediction)}

@app.post("/diabetes")
async def predict(
    pregnancies: float, glucose: float, blood_pressure: float,
    skin_thickness: float, insulin: float, bmi: float,
    diabetes_pedigree_function: float, age: float
):
    # Convert input data to numpy array
    input_data = np.array([[
        pregnancies, glucose, blood_pressure,
        skin_thickness, insulin, bmi,
        diabetes_pedigree_function, age
    ]])

    # Make predictions
    prediction = model_d2.predict(input_data)

    print(prediction)

    # Convert prediction to human-readable message
    if prediction[0] == 1:
        result = "The patient is predicted to have diabetes."
    else:
        result = "The patient is predicted to not have diabetes."

    return {"result": result, "pred_value": int(prediction[0])}

