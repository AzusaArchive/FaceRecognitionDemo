import time

import cv2
import face_recognition

cap = cv2.VideoCapture(0)

lastTime = time.time()
# 查找人脸
while True:
    # https://face-recognition.readthedocs.io/en/latest/_modules/face_recognition/api.html#face_landmarks
    ret, image = cap.read()
    # print(image.shape)
    scaled = cv2.resize(image,None,fx=0.5,fy=0.5)
    locations = face_recognition.face_locations(scaled,model="cnn")
    if any(locations):
        for loc in locations:
            print(loc[1] - loc[3], loc[2] - loc[0])
            cv2.rectangle(image,(loc[3] * 2,loc[0] * 2),(loc[1] *2,loc[2]*2),(255,255,255),2)
            pass
    t = time.time()
    frameRate = 1 / (t - lastTime)
    lastTime = t
    cv2.putText(image,str(frameRate),(500,24),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
    cv2.imshow("1",image)
    cv2.waitKey(1)
