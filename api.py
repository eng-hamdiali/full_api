from fastapi import FastAPI, HTTPException, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle 
import numpy as np
import sklearn 
from ultralytics import YOLO
import tempfile
import os
import datetime

from tensorflow.keras.models import load_model
import cv2

from fastapi.middleware.cors import CORSMiddleware




# Define the FastAPI app
app = FastAPI()

# Add CORS middleware with allow_origins set to "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Load the ML model using pickle
with open("lg.pkl", "rb") as f:
    model_d1 = pickle.load(f)


with open("db.pkl", "rb") as f:
    model_d2 = pickle.load(f)

model = load_model('skin_cancer_detector_80%.h5')

# load model
model = YOLO("dsc_c.pt")


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


@app.post("/dark_skin")
async def create_upload_file(file: UploadFile = File(...)):
    # Read image using cv2
    img_bytes = await file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # # Convert BGR to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    if img is None:
        print("Error: Unable to load image.")
    else:
        # Resize the image
        width = 640 # New width
        height = 640  # New height
        resized_image = cv2.resize(img, (width, height))
        # print(resized_image)
        # print(type(resized_image))
        # Normalize the image by dividing by 255
        # Perform object detection on the image
        results = model.predict(resized_image)
        
        for result in results:
            for box in result.boxes.data:
                x_min, y_min, x_max, y_max, conf, class_id = box
                class_name = result.names[int(class_id)]  # Get class name from index
                
                # Draw bounding box
                cv2.rectangle(resized_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (219, 105, 2), 1)

                # Add class label with confidence score
                cx = int((x_min+x_max)/2)
                cy = int((y_min+y_max)/2)
                label = f" ID: {int(class_id)} - {conf:.2f}%"  # Format label with confidence
                cv2.rectangle(resized_image, (cx-50, cy-20), (cx+110, cy+10), (219, 105, 2), -1)
                cv2.putText(resized_image, label, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                

            
                


        f_image = np.zeros((690, 640, 3), dtype=np.uint8)
        banner  = cv2.imread('psd/banner.png')

        f_image[:640, :] = resized_image
        f_image[640:, :] = banner
        
        # Get the current date and time
        current_datetime = datetime.datetime.now()
        datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")  # Format the datetime as a string
        
        # Add the date and time text to the image
        cv2.putText(f_image, datetime_str, (450, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)



        # img1 = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        # Save image
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_output.name, f_image)

        # Return FileResponse
        return FileResponse(temp_output.name, media_type='image/jpeg')
        os.unlink(temp_output.name)
# 








# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel



# # Define a Pydantic model for the request body
# class DiabetesInput(BaseModel):
#     pregnancies: float
#     glucose: float
#     blood_pressure: float
#     skin_thickness: float
#     insulin: float
#     bmi: float
#     diabetes_pedigree_function: float
#     age: float

# @app.post("/diabetes")
# async def predict(data: DiabetesInput):
#     # Convert input data to numpy array
#     input_data = np.array([[
#         data.pregnancies, data.glucose, data.blood_pressure,
#         data.skin_thickness, data.insulin, data.bmi,
#         data.diabetes_pedigree_function, data.age
#     ]])

#     # Make predictions
#     prediction = model_d2.predict(input_data)

#     print(prediction)

#     # Convert prediction to human-readable message
#     if prediction[0] == 1:
#         result = "The patient is predicted to have diabetes."
#     else:
#         result = "The patient is predicted to not have diabetes."

#     return {"result": result, "pred_value": int(prediction[0])}