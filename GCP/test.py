import tensorflow as tf
from fastapi import File, UploadFile
from tensorflow.python.lib.io import file_io
from io import BytesIO
from PIL import Image
import numpy as np


BUCKET_NAME = "malaria_catcher"
CLASS_NAMES = ["Parasitized", "Uninfected"]
model = None


def get_model(temp_model_location, bucket_model_location):
    print("###getting model")
    model_file = file_io.FileIO(bucket_model_location, mode='rb')
    print("###model downloaded")
    temp_model_file = open(temp_model_location, 'wb')
    temp_model_file.write(model_file.read())
    temp_model_file.close()
    model_file.close()
    print("###model successfully saved locally")


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        temp_model_location = '/tmp/malaria_catcher.h5'
        bucket_model_location = 'gs://malaria_catcher/models/v2_e50.h5'
        get_model(temp_model_location, bucket_model_location)

        model = tf.keras.models.load_model(temp_model_location)
        print("###model deployed")

    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    prediction = model.predict(image_batch)

    class_index = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[class_index]
    confidence = np.max(prediction[0])
    return {'class': predicted_class, 'confidence': float(confidence)}
