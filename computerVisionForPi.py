#!/usr/bin/env python3

import cv2
import numpy as np 
import cmath as m
import time

from cscore import CameraServer, VideoSource
from networktables import NetworkTablesInstance

cap = cv2.VideoCapture(0)

#source = CameraServer.getInstance().putVideo("Hello World!", cap.get(cv2.CV_CAP_PROP_FRAME_WIDTH), cap.get(cv2.CV_CAP_PROP_FRAME_HEIGHT))
source = CameraServer.getInstance().putVideo("Hello World!", int(cap.get(3)), int(cap.get(4)))
#CameraServer.getInstance().startAutomaticCapture(name="Hello World", path="/dev/video0")
while True:
    ret, img = cap.read()
    
    r = (int(img.shape[1])/int(img.shape[0]))
    graysc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    source.putFrame(graysc)
    
    ret, b = cv2.threshold(graysc, 100, 255, cv2.THRESH_BINARY)

    kernel1 = np.ones((30,30), np.uint8)
    kernel2 = np.ones((30,30), np.uint8)
    erode = cv2.erode(b, kernel1, iterations = 1)
    dilate = cv2.dilate(erode, kernel2, iterations=1)

    kernel3 = np.ones((1000,1000), np.uint8)
    superDilate = cv2.dilate(dilate, kernel3, iterations=1)

    M = cv2.moments(superDilate)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    HatchMidline = (cX, cY)

    centerX = int(img.shape[1])/2
    pDist = abs(cX-centerX)/int(img.shape[0])

    angleOfView = 60
    angleShift = int(angleOfView)*pDist

    dilateRGB = cv2.cvtColor(dilate, cv2.COLOR_GRAY2BGR)
    image=np.array(dilateRGB)
    white = np.array([255, 255, 255], np.uint8)
    notBlack = np.array([1, 1, 1], np.uint8)
    Mask = cv2.inRange(image, notBlack, white)
    pixelArea = cv2.countNonZero(Mask)

    magnitude = complex(10*17435/pixelArea)

    a = m.sin(angleShift/57.3)
    b = m.cos(angleShift/57.3)

cap.release()
