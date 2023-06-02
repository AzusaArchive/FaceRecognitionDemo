import cv2
import face_recognition

cap = cv2.VideoCapture(0)

# 查找人脸
while True:
    # https://face-recognition.readthedocs.io/en/latest/_modules/face_recognition/api.html#face_landmarks
    ret, image = cap.read()
    face_landmarks = face_recognition.face_landmarks(image)

    # 检测人脸是否打哈欠
    for landmarks in face_landmarks:
        top_lip = landmarks["top_lip"]  # 上嘴唇的点
        bottom_lip = landmarks["bottom_lip"]  # 下嘴唇的点

        top_lip_center = top_lip[9]  # 上嘴唇中心P63
        cv2.circle(image, top_lip_center, 2, (255, 0, 0), -1)
        bottom_lip_center = bottom_lip[9]  # 下嘴唇中心P67
        cv2.circle(image, bottom_lip_center, 2, (255, 0, 0), -1)
        left_lip_point = top_lip[11]  # 左嘴唇点 P60
        cv2.circle(image, left_lip_point, 2, (255, 0, 0), -1)
        right_lip_point = bottom_lip[11] # 右嘴唇点 P64
        cv2.circle(image, right_lip_point, 2, (255, 0, 0), -1)

        mouseOpenRatio = (bottom_lip_center[1] - top_lip_center[1]) / (right_lip_point[0] - left_lip_point[0])
        print(mouseOpenRatio)

        cv2.imshow("1", image)
        cv2.waitKey(1)
