import requests, json
import os
import numpy as np
import cv2 as cv
import base64

url = 'http://192.168.3.15:1234/add/'
#with open('6.jpg', 'rb') as f:
img =open('1.txt').read()

image = []
image.append(img)
res = {"image":image}

r = requests.post(url,data=res)
print(r.text)