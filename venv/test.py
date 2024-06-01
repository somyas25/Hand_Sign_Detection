import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from cvzone.ClassificationModule import Classifier
import math

# initialize the webcam (this is a constructor)
# 0 refers to the first webcam device on the system
cap = cv2.VideoCapture(0)
# initializing the hand detector and specifying to recognize only one hand in a frame
detector = HandDetector(maxHands=1)
classifier = Classifier("/Users/somyasrivastava/Desktop/Projects/Hand_Sign_Detection/Model/keras_model.h5",
                        "/Users/somyasrivastava/Desktop/Projects/Hand_Sign_Detection/Model/labels.txt")

# creating an offset for the image cropping
offset = 20
imgSize = 300
labels = ["A", "B", "C"]

folder = "/Users/somyasrivastava/Desktop/Projects/Hand_Sign_Detection/Data/C"
counter = 0

# starting an infinite loop to continuously capture frames
while True:
    # reading a frame from the webcam
    # success is a boolean which stores if the frame was captured successfully
    # img stores the frame that was captured
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        # since there is only one hand therefore we are accessing the first element of the list
        hand = hands[0]
        # we want to get the bounding box information out of the dictionary
        x, y, w, h = hand['bbox']
        # we are cropping the image using the bounding box details
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        # creating a 3D array of dimensions imgSize*imgSize and 3 color channels
        # 8 bit unsigned int is the data type of the array elements which is the standard
        # data type of images
        # we are then multiplying it with 255 to get a white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Calculating the aspect ratio to compare the size of h and w
        aspectRatio = h/w

        # if an image is taller than it is wide
        if aspectRatio > 1:
            # constant which stores the scaling factor
            k = imgSize/h
            # we multiply the scaling factor to the width to calculate the new width in relation to when
            # the height of the image would be imgSize
            # we take the upper value, and we use the ceil function
            wCal = math.ceil(k*w)
            # resize the image to the calculated values
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            # calculating the Gap such that the image sits at the center
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
            prediction, index = classifier.getPrediction(img)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal+hGap, :] = imgResize

        # getting an error if we go out of the window and the code stops !!!!
        cv2.imshow("Cropped Image", imgCrop)
        cv2.imshow("White Image", imgWhite)

    # displaying the captured frame in a window named Image
    cv2.imshow('Image', img)
    # controls the display delay timing
    cv2.waitKey(1)


