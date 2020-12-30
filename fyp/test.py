from keras.models import load_model
import cv2
import numpy as np
import sys

filepath = sys.argv[1]

REV_CLASS_MAP = {
    0:"single",
    1:"2runs",
    2:"3runs",
    3:"4runs",
    4:"5runs",
    5:"6runs" , 
    6:"none"
}


def mapper(val):
    return REV_CLASS_MAP[val]


model = load_model("rock-paper-scissors-model.h5")

# prepare the image
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (227, 227))

# predict the move made
pred = model.predict(np.array([img]))





move_code = np.argmax(pred[0])
move_name = mapper(move_code)

print("Predicted: {}".format(move_name))
