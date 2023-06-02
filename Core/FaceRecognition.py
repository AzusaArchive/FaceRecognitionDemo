from __future__ import annotations

import math
import time
from typing import Any, Literal

import cv2
import face_recognition
import numpy as np
from numpy import ndarray
from scipy.spatial import distance as dist

from Final.Utilities import Debug

# 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                         [1.330353, 7.122144, 6.903745],  # 29左眉右角
                         [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                         [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                         [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                         [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                         [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                         [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                         [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                         [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                         [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                         [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                         [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                         [0.000000, -7.415691, 4.070434]])  # 6下巴角

# 相机坐标系(XYZ)：添加相机内参
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]

# 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

# 像素坐标系(xy)：填写凸轮的本征和畸变系数
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)

dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

# 重新投影3D点的世界坐标轴以验证结果姿势
reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

# 绘制正方体12轴
line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def Log(obj):
    Debug.Log("FaceRecognition", obj)


def FindFaceLocations(imSrc, model: Literal["hog", "cnn"] = "cnn") -> list[tuple[int, int, int, int]]:
    return face_recognition.face_locations(imSrc, model=model)


def EncodeFaces(imSrc, faceLocation: list[tuple[Any, Any, Any, Any]] | None = None,
                model: Literal["hog", "cnn"] = "cnn") -> ndarray:
    if faceLocation is None:
        return face_recognition.face_encodings(imSrc, FindFaceLocations(imSrc, model))

    return face_recognition.face_encodings(imSrc, faceLocation)


def Recognize(imSrc: ndarray[Any, Any], knownFaceNames: list[str], knownFaceEncodings: list, scaleRate=0.5,
              tolerance=0.5, model: Literal["hog", "cnn"] = "cnn") -> list[str, tuple[int, int, int, int]]:
    """
    根据已知信息识别图片并返回结果
    :param model:
    :param imSrc: BGR类型的ndarray
    :param scaleRate:缩放比例，越小越快
    :param knownFaceNames:已知的人名列表
    :param knownFaceEncodings:已知的人脸编码列表
    :param tolerance: 识别阈值，0-1范围，越小越严格
    :return:返回一个列表，每个元素包含了(识别出的人脸姓名(如果未识别出则为Unknown)，(人脸上边界，右边界，下边界，左边界))
    """

    Log("*********")
    startTime = time.time()
    resizedImg = cv2.resize(imSrc, None, fx=scaleRate, fy=scaleRate)
    resizedImg = np.ascontiguousarray(resizedImg[:, :, ::-1])
    t = time.time()
    locations = FindFaceLocations(resizedImg, model)
    Log(f'Locating:{time.time() - t:.4f}s.')
    t = time.time()
    encodings = face_recognition.face_encodings(resizedImg, locations, model="large")
    Log(f'Encoding:{time.time() - t:.4f}s.')

    tf = time.time()
    matchedNames = []
    for encoding in encodings:
        name = "Unknown"
        matches = face_recognition.compare_faces(knownFaceEncodings, encoding, tolerance)
        # try:
        #     name = knownFaceNames[matches.index(True)]
        # except ValueError:
        #     pass

        # 双重验证
        face_distances = face_recognition.face_distance(knownFaceEncodings, encoding)
        if np.any(face_distances):
            best_match_index = np.argmin(face_distances)
            # print(face_distances[best_match_index])
            if matches[best_match_index]:
                name = knownFaceNames[best_match_index]

        matchedNames.append(name)
    t = time.time()
    Log(f'Compare {len(encodings)} time(s):{t - tf:.4f}s.')
    Log(f"Total time:{t - startTime:.4f}s.")
    Log("*********")
    Log("\n")

    return list(zip(matchedNames,
                    [(int(t / scaleRate), int(r / scaleRate), int(b / scaleRate), int(l / scaleRate)) for (t, r, b, l)
                     in
                     locations]))


def GetLandmarks(imSrc: ndarray, faceLocations: list[tuple[int, int, int, int]]) \
        -> list[dict[str, list[tuple[Any, Any]]]]:
    resizedImg = cv2.cvtColor(imSrc, cv2.COLOR_BGR2RGB)
    landmarks = face_recognition.face_landmarks(resizedImg, faceLocations)

    return landmarks


def GetHeadPose(landmark: dict[str, list[tuple[Any, Any]]]):
    """
    计算单个人脸的头部姿态
    :param landmark:
    :return:
    """

    # （像素坐标集合）填写2D参考点
    # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
    # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
    chin = landmark["chin"]
    leftEyeBrow = landmark["left_eyebrow"]
    rightEyeBrow = landmark["right_eyebrow"]
    leftEye = landmark["left_eye"]
    rightEye = landmark["right_eye"]
    noseTip = landmark["nose_tip"]
    topLip = landmark["top_lip"]
    bottomLip = landmark["bottom_lip"]
    P17 = leftEyeBrow[0]
    P21 = leftEyeBrow[4]
    P22 = rightEyeBrow[0]
    P26 = rightEyeBrow[4]
    P36 = leftEye[0]
    P39 = leftEye[3]
    P42 = rightEye[0]
    P45 = rightEye[3]
    P31 = noseTip[0]
    P35 = noseTip[4]
    P48 = topLip[0]
    P54 = topLip[6]
    P57 = bottomLip[3]
    P8 = chin[8]

    points = [P17, P21, P22, P26, P36, P39, P42, P45, P31, P35, P48, P54, P57, P8]
    image_pts = np.array([list(pt) for pt in points], dtype=np.float32)

    # solvePnP计算姿势——求解旋转和平移矩阵：
    # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

    # 计算欧拉角calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
    # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    # pitch, yaw, roll = [math.radians(_) for _ in euler_angle]
    #
    # pitch = math.degrees(math.asin(math.sin(pitch)))
    # roll = -math.degrees(math.asin(math.sin(roll)))
    # yaw = math.degrees(math.asin(math.sin(yaw)))
    # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    return reprojectdst, euler_angle  # 投影误差，欧拉角


def IsEyesClosed(landmark: dict[str, list[tuple[Any, Any]]], threshold: float = 0.18) -> bool:
    """
    判断是否闭眼
    :param landmark:特征点字典
    :param threshold:检测阈值，双眼各自长宽比低于此值则判定闭眼
    :return:
    """
    leftEye = landmark["left_eye"]
    rightEye = landmark["right_eye"]

    A1 = dist.euclidean(leftEye[1], leftEye[5])  # P37 P41
    B1 = dist.euclidean(leftEye[2], leftEye[4])  # P38 P40
    C1 = dist.euclidean(leftEye[0], leftEye[3])  # P36 P39

    A2 = dist.euclidean(rightEye[1], rightEye[5])  # P43 P47
    B2 = dist.euclidean(rightEye[2], rightEye[4])  # P44 P46
    C2 = dist.euclidean(rightEye[0], rightEye[3])  # P42 P45

    le = (A1 + B1) / (2.0 * C1)
    re = (A2 + B2) / (2.0 * C2)

    return le < threshold and re < threshold


def IsEyesClosed2(landmark: dict[str, list[tuple[Any, Any,]]], threshold: float = 5.0) -> bool:
    def MyEyeDist(pt1, pt2):
        return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    leftEye = landmark["left_eye"]
    rightEye = landmark["right_eye"]

    lt = MyEyeDist(leftEye[3], leftEye[0]) / MyEyeDist(leftEye[4], leftEye[2])
    rt = MyEyeDist(rightEye[3], rightEye[0]) / MyEyeDist(rightEye[4], rightEye[2])

    return lt > threshold and rt > threshold


def IsMouseOpened(landmark: dict[str, list[tuple[Any, Any]]], threshold: float = 0.5) -> bool:
    """
    判断是否张嘴
    :param landmark:特征点字典
    :param threshold:检测阈值，上下和左右嘴唇点比例大于此值则判定张嘴
    :return:
    """
    top_lip = landmark["top_lip"]  # 上嘴唇的点
    bottom_lip = landmark["bottom_lip"]  # 下嘴唇的点

    top_lip_center = top_lip[9]  # 上嘴唇中心P63
    bottom_lip_center = bottom_lip[9]  # 下嘴唇中心P67
    left_lip_point = top_lip[11]  # 左嘴唇点 P60
    right_lip_point = bottom_lip[11]  # 右嘴唇点 P64

    mouseOpenRatio = (bottom_lip_center[1] - top_lip_center[1]) / (right_lip_point[0] - left_lip_point[0])
    return mouseOpenRatio > threshold


def IsHeadDown(eulerAngle: ndarray[ndarray[np.float32]], threshold=5):
    return eulerAngle[0][0] > threshold  # 俯仰轴


def DrawFaceRect(imSrc: ndarray, location: tuple[int, int, int, int],
                 color: tuple[int, int, int] = (255, 255, 255)) -> None:
    """
    在图片上绘制人脸的矩形框和姓名
    :param color:
    :param imSrc:源图像
    :param location:矩形框位置
    :return:绘制后的图片
    """

    top, right, bottom, left = location
    cv2.rectangle(imSrc, (left, top), (right, bottom), color, 1)


def PutText(imSrc: ndarray, text: str, location: tuple[int, int], fontScale: float = 0.6,
            fontColor: tuple[int, int, int] = (255, 255, 255)) -> None:
    font = cv2.FONT_ITALIC
    cv2.putText(imSrc, text, location, font, fontScale, fontColor, 1)


def DrawLandmarks(imSrc: ndarray, landmarks: list[dict[str, list[tuple[Any, Any]]]]) -> None:
    """
    在图片上绘制多个人脸的轮廓
    :param imSrc:图像源，BGR
    :param landmarks:人脸特征点列表
    :return:
    """
    for landmark in landmarks:
        for part in landmark.values():
            for mark in part:
                cv2.circle(imSrc, mark, 2, (255, 245, 163), -1)


def DrawHeadPose(imSrc: ndarray, reprojectDist: ndarray):
    for start, end in line_pairs:
        p1 = reprojectDist[start]
        p2 = reprojectDist[end]
        cv2.line(imSrc, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255))


def MainTest():
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        print(img.shape)
        for name, location in Recognize(img, [], []):
            DrawFaceRect(img, location)
            PutText(img, name, (location[3] + 6, location[2] - 6))
        cv2.imshow("1", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    MainTest()
