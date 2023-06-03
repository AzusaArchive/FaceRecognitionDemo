from __future__ import annotations

import asyncio
import tkinter as tk
import tkinter.messagebox
import traceback
from typing import Optional

from numpy import ndarray

import Final.Utilities.FaceDataRepository as FaceData
import Final.Core.FrameCapture as Frame
import Final.Core.FaceRecognition as FaceRec


class FaceInputPage:
    def __init__(self, parentWindow: tk.Tk, loop: asyncio.AbstractEventLoop) -> None:
        self.parent = parentWindow
        self.frameCapture: Optional[Frame] = None

        windowWidth = 960
        windowHeight = 540

        self.window = tk.Toplevel(self.parent, bg="#1E1E1E")
        self.window.title("input face")
        self.window.geometry(f"{windowWidth}x{windowHeight}")
        self.window.iconbitmap("Assets/Images/Icon.ico")
        # self.window.protocol("WM_DELETE_WINDOW", lambda: (self.frameCapture.Close(), self.window.destroy()))

        self.display = tk.Label(self.window, bg="#1E1E1E")
        self.display.place(x=0, y=0, width=windowWidth, height=windowHeight)

        self.header = tk.Frame(self.window, height=30, bg="#323233")
        self.header.place(x=0, y=0, width=windowWidth)
        self.btn_confirm = tk.Button(self.header, text="确认录入", fg="#ffffff", bg="#323233", width=16, relief=tk.FLAT,
                                     command=self.ConfirmFaceData)
        self.btn_confirm.grid(row=0, column=0, sticky="NSEW")

        self.task = loop.create_task(self.StartRecord())
        self.currentFrame = None

    async def StartRecord(self):
        try:
            with await Frame.FromCamera() as self.frameCapture:
                while True:
                    ret, frame = self.frameCapture.GetFrame()
                    if not ret:
                        raise "无法连接到摄像头"
                    if frame is not None:
                        self.currentFrame = frame
                        result = FaceRec.Recognize(frame, list(FaceData.faceData.keys()), list(FaceData.faceData.values()))
                        for name, location in result:
                            FaceRec.DrawFaceRect(frame, location)
                        tkImg = Frame.BGRArray2TkImg(frame)
                        self.display.config(image=tkImg)
                        self.display.image = tkImg
                    await asyncio.sleep(0.033)
        except Exception:
            traceback.print_exc()
            raise

    def ConfirmFaceData(self):
        if self.currentFrame is None:
            return
        encodings = FaceRec.EncodeFaces(self.currentFrame)
        if tk.messagebox.askyesno("图像获取完成", "确认保存此人脸数据？"):
            inputWindow = tk.Toplevel(self.window)
            inputWindow.geometry("400x160")
            inputWindow.iconbitmap("Assets/Images/Icon.ico")
            inputWindow.title("输入人名")
            tk.Label(inputWindow, text="请输入人名").pack(padx=45, pady=(25, 5), anchor="w")
            inputField = tk.Entry(inputWindow, width=80)
            inputField.pack(padx=45, pady=5, anchor="w")
            btn_confirm = tk.Button(inputWindow, relief=tk.GROOVE, width=45, text="确认",
                                    command=lambda:
                                    (
                                        self.SaveFaceData(inputField.get(), encodings),
                                        self.task.cancel(),
                                        inputWindow.destroy(),
                                        self.window.destroy()
                                    ))
            btn_confirm.pack(anchor="center", padx=65, pady=25)

    @staticmethod
    def SaveFaceData(name: str, encodings: ndarray):
        if len(encodings) > 1:
            tk.messagebox.showerror(message="图像中包含多个人脸，请重试")
            return
        FaceData.SaveFaceData(name, encodings[0].tolist())
        tk.messagebox.showinfo(message=f"已保存{name}的人脸信息")
