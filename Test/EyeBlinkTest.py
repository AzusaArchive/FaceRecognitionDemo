import cv2

from Final.Core import FaceRecognition as FaceRec

cap = cv2.VideoCapture(0)
eyeClosed = False
eyeBlinkTimes = 0
while True:
    ret, img = cap.read()
    locations = FaceRec.FindFaceLocations(img)
    landmarks = FaceRec.GetLandmarks(img,locations)
    if any(landmarks):
        if FaceRec.IsEyesClosed2(landmarks[0]):
            eyeClosed = True
        else:
            if eyeClosed:
                eyeBlinkTimes += 1
                print(f"Eye blink times: {eyeBlinkTimes}")
            eyeClosed = False
        FaceRec.DrawFaceRect(img,locations[0])
        FaceRec.DrawLandmarks(img,landmarks)
    cv2.imshow("eye blink detection", img)

    cv2.waitKey(1)