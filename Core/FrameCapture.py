from __future__ import annotations

import asyncio
from typing import Optional

import cv2
from PIL import Image, ImageTk
from cv2 import VideoCapture
from numpy import ndarray

cap: Optional[Frame] = None


async def FromCamera() -> Frame:
    global cap
    if cap is None:
        cap = Frame()
    await asyncio.sleep(0.1)  # 小等一下IO，让tk能够更新一次标签
    cap.OpenCamera()
    return cap


async def FromVideo(path: str) -> Frame:
    global cap
    if cap is None:
        cap = Frame()
    await asyncio.sleep(0.1)
    cap.OpenVideo(path)
    return cap

def BGRArray2PILImg(bgrArr: ndarray) -> Image.Image:
    """
    将BGR格式的图片数组转换为PIL所用的图像
    :param bgrArr:
    :return:
    """
    rgbFrame = cv2.cvtColor(bgrArr, cv2.COLOR_BGR2RGBA)
    pilFrame = Image.fromarray(rgbFrame)
    return pilFrame


def BGRArray2TkImg(bgrArr: ndarray) -> ImageTk.PhotoImage:
    """
    将BGR格式的图片数组转换为tk所用的图像
    :param bgrArr:
    :return:
    """
    rgb = cv2.cvtColor(bgrArr, cv2.COLOR_BGR2RGBA)
    pil = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil)


class Frame:
    __cap: VideoCapture | None

    def __init__(self):
        self.__cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not all([exc_type, exc_val, exc_tb]):
            print("Exception raised in FrameCapture:")
            print(f"\ttype: {exc_type}")
            print(f"\tval: {exc_val}")
            print(f"\ttb: {exc_tb}")
        self.__cap.release()

    def OpenCamera(self):
        """
        打开摄像头，严重阻塞线程
        :return:
        """
        self.__cap = cv2.VideoCapture(0)

    def OpenVideo(self, path: str):
        self.__cap = cv2.VideoCapture(path)

    def IsActive(self) -> bool:
        return self.__cap is not None and self.__cap.isOpened()

    def Close(self):
        if self.__cap is not None:
            self.__cap.release()
            print("视频截取关闭")

    def GetFrame(self, scale=1.0) -> tuple[bool, ndarray]:
        if self.__cap:
            ret, frame = self.__cap.read()
            if ret:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
                return ret, frame
            return ret, frame
        else:
            raise "未初始化"

    def GetFrameRate(self) -> float:
        return self.__cap.get(cv2.CAP_PROP_FPS)

