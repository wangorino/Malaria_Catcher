from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

MODEL = tf.keras.models.load_model("E:/PycharmProjects/MalariaCatcher/models/v2_e50.h5")
CLASS_NAMES = ["Parasitized", "Uninfected"]

@app.get("/test")
def test():
    return "Finally..."

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)

    prediction = MODEL.predict(image_batch)

    class_index = np.argmax(prediction[0])
    predicted_class = CLASS_NAMES[class_index]
    confidence = np.max(prediction[0])
    return {'class': predicted_class, 'confidence': float(confidence)}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
