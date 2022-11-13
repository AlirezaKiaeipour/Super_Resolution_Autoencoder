import argparse
import os
import random
import numpy as np
import cv2
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument("--input_model",type=str,help="Please Enter path of input model/  Example: model.h5")
parser.add_argument("--input_image",type=str,help="Please Enter path of input image/  Example: image.jpg")
arg = parser.parse_args()
num = random.randint(1,5000)
model = load_model(arg.input_model)
os.makedirs("img",exist_ok=True)

image = cv2.imread(arg.input_image)
image = cv2.resize(image,(256,256))
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = image / 255.0
image = np.expand_dims(image,axis=0)
pred = model.predict(image)
pred = pred * 255.0
pred = cv2.cvtColor(pred[0],cv2.COLOR_RGB2BGR)
cv2.imwrite(f"img/High_Resolution_{num}.jpg",pred)
