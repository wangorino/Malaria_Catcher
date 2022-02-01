import tensorflow as tf
from tensorflow.python.lib.io import file_io


def get_model(self):
    print("###getting model")
    model_file = file_io.FileIO('gs://malaria_catcher/models/3.h5', mode='rb')
    print("###model downloaded")
    temp_model_location = '/tmp/malaria_catcher.h5'
    temp_model_file = open(temp_model_location, 'wb')
    temp_model_file.write(model_file.read())
    temp_model_file.close()
    model_file.close()
    print("###model successfully saved locally")

    model = tf.keras.models.load_model(temp_model_location)
    print("###model deployed")
    return {"status": "success"}
