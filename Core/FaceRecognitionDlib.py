from __future__ import annotations

from scipy.spatial import distance as dist
from typing import Any

import cv2
import dlib
from imutils import face_utils
from numpy import ndarray
import numpy as np

from Final.Utilities import Debug

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Assets/shape_predictor_68_face_landmarks.dat")
encoder = dlib.face_recognition_model_v1("Assets/dlib_face_recognition_resnet_model_v1.dat")

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
    Debug.Log("FaceRecognitionDlib", obj)


def FindFaceLocations(imSrc) -> list[tuple[int, int, int, int]]:
    locations = detector(imSrc)
    print(locations)
    return locations

def EncodeFaces(imSrc:ndarray, faceLocations) -> ndarray:
    raise


def Recognize(imSrc:ndarray, knownFaceName:list[str], knownFaceEncodings:list, scaleRate:float = 0.5, tolerance:float = 0.5):
    locations= FindFaceLocations(imSrc)
    shapes = []
    for loc in locations:
        shapes.append(predictor(imSrc,loc))
    landmarks = GetLandMarks(imSrc,locations)
    raise



def GetLandMarks(imSrc: ndarray, faceLocations: list[tuple[int, int, int, int]]) -> ndarray[ndarray[ndarray[np.int32]]]:
    """

    :param imSrc:
    :param faceLocations:
    :return: 每个人脸对应特征点坐标的numpy数组
    """
    landmarks = []
    for location in faceLocations:
        landmarks.append(face_utils.shape_to_np(predictor(imSrc, location)))
    return landmarks


def GetHeadPose(imSrc: ndarray) -> tuple[Any, Any]:
    # 填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
    """
      17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
      45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
    """
    # 像素坐标集合
    image_pts = np.float32([imSrc[17], imSrc[21], imSrc[22], imSrc[26], imSrc[36],
                            imSrc[39], imSrc[42], imSrc[45], imSrc[31], imSrc[35],
                            imSrc[48], imSrc[54], imSrc[57], imSrc[8]])
    """
    用solvepnp或sovlepnpRansac，输入3d点、2d点、相机内参、相机畸变，输出r、t之后
    用projectPoints，输入3d点、相机内参、相机畸变、r、t，输出重投影2d点
    计算原2d点和重投影2d点的距离作为重投影误差
    """
    # solvePnP计算姿势——求解旋转和平移矩阵：
    # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)
    # projectPoints重新投影误差
    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示

    # 计算欧拉角calc euler angle
    # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
    # eulerAngles –可选的三元素矢量，包含三个以度为单位的欧拉旋转角度
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)  # 将投影矩阵分解为旋转矩阵和相机矩阵

    return reprojectdst, euler_angle


def IsEyesClosed(landmark: ndarray[ndarray[np.int32]], threshold: float = 0.2):
    A1 = dist.euclidean(landmark[37], landmark[41])  # P37 P41
    B1 = dist.euclidean(landmark[38], landmark[40])  # P38 P40
    C1 = dist.euclidean(landmark[36], landmark[39])  # P36 P39

    A2 = dist.euclidean(landmark[43], landmark[47])  # P43 P47
    B2 = dist.euclidean(landmark[44], landmark[46])  # P44 P46
    C2 = dist.euclidean(landmark[42], landmark[45])  # P42 P45

    le = (A1 + B1) / (2.0 * C1)
    re = (A2 + B2) / (2.0 * C2)

    return le < threshold and re < threshold


def IsMouseOpened(landmark: ndarray[ndarray[np.int32]], threshold: float = 0.5) -> bool:
    top_lip_center = landmark[63]  # 上嘴唇中心P63
    bottom_lip_center = landmark[67]  # 下嘴唇中心P67
    left_lip_point = landmark[60]  # 左嘴唇点 P60
    right_lip_point = landmark[64]  # 右嘴唇点 P64

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
    # TODO: 添加cv2.putText中文支持

    top, right, bottom, left = location
    cv2.rectangle(imSrc, (left, top), (right, bottom), color, 1)


def PutText(imSrc: ndarray, text: str, location: tuple[int, int], fontScale: float = 0.6,
            fontColor: tuple[int, int, int] = (255, 255, 255)) -> None:
    font = cv2.FONT_ITALIC
    cv2.putText(imSrc, text, location, font, fontScale, fontColor, 1)


def DrawLandmarks(imSrc: ndarray, landmarks: ndarray):
    for landmark in landmarks:
        for point in landmark:
            cv2.circle(imSrc, (point[0], point[1]), 2, (255, 245, 163), -1)


def DrawHeadPose(imSrc: ndarray, reprojectDist: ndarray):
    for start, end in line_pairs:
        p1 = reprojectDist[start]
        p2 = reprojectDist[end]
        cv2.line(imSrc, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255))

