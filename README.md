# Darknet-Tenserflow

Installing darkflow on Ubuntu 16.04
Installing  Darknet or Darkflow is very easy task.

Its an opensource project that was developed un YOLO the real time object detection model.

Requirements:

1. Python 3.5 or later
2. tensorflow
3. OPencv


Note:   Installing  tensorflow

code@code:~$ sudo pip3 install tensorflow 


step 1:  Download darkflow

git clone https://github.com/thtrieu/darkflow.git

step 2 :

cd  darkflow
 python3  setup.py  build_ext --inplace

or

python3  install -e .

########### to detect object from ImAGES ###########

./darknet detect cfg/yolo.cfg  yolo.weights    dog.jpg


###### USING darflow from another python Application #####

from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("./test/dog.jpg")
result = tfnet.return_predict(imgcv)
print(result)


