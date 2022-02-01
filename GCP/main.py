import tensorflow as tf
from tensorflow.python.lib.io import file_io
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

def predict(request):
    global model
    if model is None:
        temp_model_location = '/tmp/malaria_catcher.h5'
        bucket_model_location = 'gs://malaria_catcher/models/v2_e50.h5'
        get_model(temp_model_location, bucket_model_location)

        model = tf.keras.models.load_model(temp_model_location)
        print("###model deployed")

    print("###getting image")
    image = request.files["file"]
    image = np.array(
        Image.open(image).convert("RGB").resize((128, 128))
    )
    print(image.shape)
    image = image/255
    image_array = tf.expand_dims(image, 0)
    print("###preprcessing finished")

    prediction = model.predict(image_array)

    class_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[class_index]
    confidence = np.max(prediction)
    return {'class': predicted_class, 'confidence': float(confidence)}






