from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import tempfile
import os

# Load model
model = load_model('skin_cancer_detector_80%.h5')

app = FastAPI()

@app.post("/uploadfile_to_text")
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
    

@app.post("/uploadfile_to_image")
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
        # print(resized_image)
        # print(type(resized_image))
        # Normalize the image by dividing by 255
        
        normalized_image = np.asarray(resized_image / 255.0)

        # Convert RGB to BGR
        
        normalized_image_reshaped = np.expand_dims(normalized_image, axis=0)
        # print(normalized_image_reshaped)
        result = model.predict(normalized_image_reshaped)

        # Existing dictionary
        existing_dict = {
            0: 'actinic keratoses & ic',
            1: 'Basal cell',
            2: 'Benign keratosis',
            3: 'Dermatofibroma',
            4: 'Melanoma',
            5: 'Melanocytic nevi',
            6: 'Vascular skin'
        }

        # Create a new dictionary
        new_list = []
        max_class_index = np.argmax(result[0])
        for i, value in enumerate(result[0]):
            class_name = existing_dict[i]
            probability = '{:.5f}'.format(value)
            new_list.append(
                (class_name,probability))

        new_list = sorted(new_list, key=lambda x: x[1], reverse=True)

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_position = (5, 30)
        text_color = (0,128,0)
        thickness = 1
        font_scale = 0.6
        
        first_text= '1- ' + new_list[0][0]+' : '+ new_list[0][1]
        second_text = '2- ' + new_list[1][0]+' : '+ new_list[1][1]
        cv2.putText(img, 'Skin Diseases Propability : ', text_position, font, 0.8, text_color, 2, cv2.LINE_AA)
        cv2.putText(img, first_text, (8, 60), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(img, second_text, (8, 80), font, font_scale, text_color, thickness, cv2.LINE_AA)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Save image
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_output.name, img)

    # Return FileResponse
    return FileResponse(temp_output.name, media_type='image/jpeg')
    os.unlink(temp_output.name)
