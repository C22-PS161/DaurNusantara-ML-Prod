from flask import Flask, request
from base64 import decode
from matplotlib import pyplot as plt

import os
import cv2 
import numpy as np
import tensorflow as tf

app = Flask(__name__)
SAVED_MODEL_PATH = "./savedgraphmodel/saved_model"
IMAGE_PATH = "./test.png"

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

def get_results(detections, threshold):
  last_valid_idx = 0
  while (detections['detection_scores'][last_valid_idx] >= threshold):
    last_valid_idx += 1
  
  used_scores = detections['detection_scores'][:last_valid_idx]
  used_classes = detections['detection_classes'][:last_valid_idx]
  return used_scores, used_classes

@app.route('/', methods=['POST'])
def predict():
    threshold = float(request.form['threshold'])
    request_img = request.files['img'].read()
    #convert string data to numpy array
    npimg = np.fromstring(request_img, np.uint8)

# convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    images_np = np.array(img)

    # change file image into numpy array
    input_tensor = tf.convert_to_tensor(np.expand_dims(images_np, 0), dtype=tf.uint8)

    detections = f(input_tensor)
    detections['detection_classes'] = detections['detection_classes'].numpy().astype(np.int64)[0]
    detections['detection_scores'] = detections['detection_scores'][0].numpy()

    used_scores, used_classes = get_results(detections, threshold)
    print(used_classes)

    decoded_classes = []
    for i in range (len(used_classes)):
      decoded_classes.append(label_mapping['{}'.format(used_classes[i])])
    
    # return json, objects is list of string
    return {
      "objects" : decoded_classes
      }

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))