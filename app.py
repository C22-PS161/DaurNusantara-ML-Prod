from flask import Flask
from matplotlib import pyplot as plt

import cv2 
import numpy as np
import tensorflow as tf

app = Flask(__name__)
SAVED_MODEL_PATH = "./savedgraphmodel/saved_model"
IMAGE_PATH = "./test.jpg"

label_mapping = {
    '1': 'kantong',
    '2': 'kertas',
    '3': 'piring',
    '4': 'kardus',
    '5': 'cup',
    '6': 'kaleng',
    '7': 'botol'
    }

imported = tf.saved_model.load(SAVED_MODEL_PATH)
f = imported.signatures['serving_default']

@app.route('/')
def predict():
    #get image by http-get
    image_from_api = IMAGE_PATH
    img = cv2.imread(image_from_api)
    images_np = np.array(img)

    # change file image into numpy array
    input_tensor = tf.convert_to_tensor(np.expand_dims(images_np, 0), dtype=tf.uint8)

    detections = f(input_tensor)
    detections['detection_classes'] = detections['detection_classes'].numpy().astype(np.int64)[0]
    detections['detection_scores'] = detections['detection_scores'][0].numpy()

    hi_acc_idx = detections['detection_scores'].argmax()
    int_label = detections['detection_classes'][hi_acc_idx]
    
    return label_mapping['{}'.format(int_label)]

if __name__ == '__main__':
    app.run(host='127.0.0.1')