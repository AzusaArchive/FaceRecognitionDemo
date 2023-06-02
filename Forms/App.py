import asyncio
import time
import tkinter as tk
import tkinter.filedialog as TkFileDiag
import tkinter.ttk as ttk
import traceback
from asyncio import Task, AbstractEventLoop
from typing import Literal, Optional, Any

import Final.Utilities.FaceDataRepository as FaceData
import Final.Core.FaceRecognition as FaceRec
import Final.Core.FrameCapture as Frame
from Final.Forms.FaceInputPage import FaceInputPage
from Final.Utilities import Debug
from Final.Entities.DetectionInfo import DetectionInfo


class App:
    def __init__(self, loop: AbstractEventLoop) -> None:
        self.window = tk.Tk()
        self.window.title("Main")
        self.window.config(bg="#1E1E1E")
        windowWidth = 1280
        windowHeight = 780
        self.window.geometry(f"{windowWidth}x{windowHeight}")
        self.window.iconbitmap("Assets/Images/Icon.ico")

        # 初始化显示区域
        self.label_display = tk.Label(self.window, bg="#000000", fg="#DDDDDD")
        self.label_display.place(x=0, y=0, relheight=1, relwidth=3 / 4)
        self.frameRateVar: tk.StringVar = tk.StringVar(value="FPS")
        self.label_frameRate = tk.Label(self.label_display, height=2, width=6, textvariable=self.frameRateVar,
                                        fg="#DDDDDD", bg="#1E1E1E")
        self.label_frameRate.place(x=0, y=30, relx=1, anchor="ne")

        # ttk 样式
        ttkStyle = ttk.Style()
        ttkStyle.theme_use("alt")
        ttkStyle.configure("black.Treeview", background="#1E1E1E", foreground="#DDDDDD", fieldbackground="#1E1E1E",
                           rowheight=25)
        ttkStyle.configure("black.Treeview.Heading", background="#1E1E1E", foreground="#DDDDDD")
        ttkStyle.configure("black.Vertical.TScrollbar", background="#444444", troughcolor="#1E1E1E", arrowsize=8,
                           width=8)

        # 初始化姓名板
        self.nameBoard = tk.Frame(self.window, bg="#1E1E1E", relief=tk.RIDGE, bd=1)
        self.nameBoard.place(x=0, y=30, relx=1, anchor="ne", relheight=(windowHeight - 60) / windowHeight / 2,
                             relwidth=1 / 4)
        self.nameTable: Optional[ttk.Treeview] = None
        self.InitNameTable()

        # 初始化参数控制区域
        self.imageScaleVar = tk.DoubleVar(value=1)
        self.locatingModelVar = tk.IntVar(value=1)
        self.faceRecToleranceVar = tk.DoubleVar(value=0.45)
        self.eyeClosedThresholdVar = tk.DoubleVar(value=0.2)
        self.mouseOpenedThresholdVar = tk.DoubleVar(value=0.5)
        self.expressionCheckFrequencyVar = tk.IntVar(value=30)
        self.asleepTimeThresholdVar = tk.DoubleVar(value=60)
        self.headDownAngleThresholdVar = tk.IntVar(value=10)

        self.controlPanelWrapper: tk.Canvas = tk.Canvas(self.window, bg="#1E1E1E", relief=tk.RIDGE, bd=1,
                                                        highlightthickness=0)
        self.controlPanelWrapper.place(x=0, y=-30, relx=1, rely=1, anchor="se",
                                       relheight=(windowHeight - 60) / windowHeight / 2,
                                       relwidth=1 / 4)
        self.InitControlPanel()

        # 初始化页头
        self.header = tk.Frame(self.window, height=30, bg="#323233", width=windowWidth)
        self.header.rowconfigure(0, weight=1)
        self.header.place(x=0, y=0, height=30, relwidth=1)

        self.btn_openVideo = tk.Button(self.header, text="打开视频文件", fg="#DDDDDD", bg="#323233", width=16,
                                       activebackground="#646466", relief=tk.FLAT,
                                       command=lambda: (
                                           self.SelectVideo(),
                                           self.SwitchDetectionMode("Video")
                                       ))
        self.btn_openVideo.grid(row=0, column=0)

        self.btn_openCamera = tk.Button(self.header, text="开启摄像头", fg="#DDDDDD", bg="#323233", width=16,
                                        activebackground="#646466", relief=tk.FLAT,
                                        command=lambda: self.SwitchDetectionMode("Camera"))
        self.btn_openCamera.grid(row=0, column=1)

        self.btn_inputFace = tk.Button(self.header, text="录入人脸", fg="#DDDDDD", bg="#323233", width=16,
                                       relief=tk.FLAT,
                                       command=lambda: (
                                           self.SwitchDetectionMode("None"),
                                           FaceInputPage(self.window, self.eventLoop))
                                       )
        self.btn_inputFace.grid(row=0, column=2)

        self.btn_closeFrame = tk.Button(self.header, text="关闭视频", fg="#DDDDDD", bg="#323233", width=16,
                                        activebackground="#646466", relief=tk.FLAT,
                                        command=lambda: self.SwitchDetectionMode("None"))
        self.btn_closeFrame.grid(row=0, column=3)

        # 初始化页尾
        self.footer = tk.Frame(self.window, height=30, bg="#323233", width=windowWidth)
        self.footer.rowconfigure(0, weight=1)
        self.footer.place(x=0, y=0, rely=1, height=30, relwidth=1, anchor="sw")

        self.hint = tk.StringVar(value="请选择文件或开启摄像头")
        self.label_hint = tk.Label(self.footer, textvariable=self.hint, fg="#DDDDDD", bg="#323233")
        self.label_hint.grid(row=0, column=0, sticky="NSEW", padx=20)

        self.drawLandmarkVar: tk.BooleanVar = tk.BooleanVar(value=True)
        self.drawFaceRectVar: tk.BooleanVar = tk.BooleanVar(value=True)
        self.drawHeadPose: tk.BooleanVar = tk.BooleanVar(value=True)
        self.checkBox_enableLandmark = tk.Checkbutton(self.footer, text="绘制面部特征点", bg="#323233", fg="#DDDDDD",
                                                      selectcolor="#323233", activebackground="#646466",
                                                      activeforeground="#DDDDDD", variable=self.drawLandmarkVar)
        self.checkBox_enableLandmark.place(x=-20, y=0, relx=1, anchor="ne")
        self.checkBox_enableFaceRect = tk.Checkbutton(self.footer, text="绘制人脸矩形框", bg="#323233",
                                                      fg="#DDDDDD",
                                                      selectcolor="#323233", activebackground="#646466",
                                                      activeforeground="#DDDDDD", variable=self.drawFaceRectVar)
        self.checkBox_enableFaceRect.place(x=-140, y=0, relx=1, anchor="ne")
        self.checkBox_enableFaceRect = tk.Checkbutton(self.footer, text="绘制头部姿态轮廓", bg="#323233",
                                                      fg="#DDDDDD",
                                                      selectcolor="#323233", activebackground="#646466",
                                                      activeforeground="#DDDDDD", variable=self.drawHeadPose)
        self.checkBox_enableFaceRect.place(x=-260, y=0, relx=1, anchor="ne")

        # 字段
        # 识别数据
        self.videoPath: str = ""
        self.frameCapture: Optional[Frame.Frame] = None
        self.timestamp_display: float = time.time()
        self.currentFrame = None
        self.faceData: Optional[list[tuple[str, tuple[int, int, int, int]]]] = None
        self.landmarks: Optional[list[dict[str, list[tuple[Any, Any]]]]] = None
        self.detectionData: Optional[dict[str, DetectionInfo]] = dict()
        self.headPoses: Optional[list[tuple[Any, Any]]] = None

        # 协程控制
        self.eventLoop = loop
        self.task_recognition: Optional[Task] = None
        self.task_updateNameBoard: Optional[Task] = None

    def InitNameTable(self):
        self.nameTable = ttk.Treeview(self.nameBoard, columns=["#", "name", "status", "blink_times", "online_time"],
                                      show="headings", style="black.Treeview", height=45)
        self.nameTable.pack(fill="both", expand=True)

        nameTableScroll = ttk.Scrollbar(self.nameTable, style="black.Vertical.TScrollbar", command=self.nameTable.yview)
        nameTableScroll.pack(side='right', fill='y')
        self.nameTable.config(yscrollcommand=nameTableScroll.set)

        self.nameTable.tag_configure("black", background="#1E1E1E", foreground="#DDDDDD")

        self.nameTable.column("#", width=10, anchor=tk.CENTER)
        self.nameTable.column("name", width=50, anchor=tk.CENTER)
        self.nameTable.column("status", width=40, anchor=tk.CENTER)
        self.nameTable.column("blink_times", width=50, anchor=tk.CENTER)
        self.nameTable.column("online_time", width=60, anchor=tk.CENTER)

        self.nameTable.heading("#", text="#")
        self.nameTable.heading("name", text="姓名")
        self.nameTable.heading("status", text="状态")
        self.nameTable.heading("blink_times", text="眨眼次数")
        self.nameTable.heading("online_time", text="在线时间")
        #
        # self.nameTable.insert("", "end", values=["1","x","y","z","11:11"])
        # self.nameTable.insert("", "end", values=["1","x","y","z","11:11"])
        # self.nameTable.insert("", "end", values=["1","x","y","z","11:11"])

    def InitControlPanel(self):

        controlPanel = tk.Frame(self.controlPanelWrapper, bg="#1E1E1E")
        controlPanel.pack(side="left", fill="both", expand=True)
        # 绑定滚动区域
        controlPanel.bind("<Configure>", lambda event: self.controlPanelWrapper.config(
            scrollregion=self.controlPanelWrapper.bbox("all")))
        # 绑定鼠标滚轮事件
        controlPanel.bind_all("<MouseWheel>",
                              lambda event: self.controlPanelWrapper.yview_scroll(int(-1 * (event.delta / 120)),
                                                                                  "units"))

        scroll = ttk.Scrollbar(self.controlPanelWrapper, command=self.controlPanelWrapper.yview,
                               style="black.Vertical.TScrollbar")
        scroll.pack(side="right", fill="y")
        self.controlPanelWrapper.config(yscrollcommand=scroll.set)

        self.controlPanelWrapper.create_window((5, 5), window=controlPanel, anchor="nw")

        # TODO:随父级窗口变化
        slider_imageScale = tk.Scale(controlPanel, bg="#1E1E1E", fg="#DDDDDD",
                                     label="图像缩放率：", orient=tk.HORIZONTAL, relief=tk.FLAT,
                                     bd=1, from_=0.1, to=1, resolution=0.01,
                                     highlightthickness=0, sliderrelief=tk.FLAT, troughcolor="gray",
                                     activebackground="#323233", width="6", sliderlength="20", length=270,
                                     variable=self.imageScaleVar)
        slider_imageScale.grid(column=0, padx=12, pady=5, sticky="nw")
        tk.Label(controlPanel, text="调整图像的缩放比例，降低以提高性能。（默认：1）", bg="#1E1E1E", fg="#DDDDDD") \
            .grid(column=0, padx=12, pady=(0, 24), sticky="nw")

        tk.Label(controlPanel, text="人脸定位模型：", bg="#1E1E1E", fg="#DDDDDD") \
            .grid(column=0, padx=12, sticky="nw")
        frame_locatingModel = tk.Frame(controlPanel, bg="#1E1E1E")
        frame_locatingModel.grid(column=0, padx=12, pady=(0,24), sticky="nw")
        tk.Radiobutton(frame_locatingModel, text="hog", bg="#1E1E1E", fg="#DDDDDD",selectcolor="#1E1E1E",
                       value=0, variable=self.locatingModelVar,).pack(side="left", padx=12)
        tk.Radiobutton(frame_locatingModel, text="cnn", bg="#1E1E1E", fg="#DDDDDD",selectcolor="#1E1E1E",
                       value=1, variable=self.locatingModelVar).pack(side="left", padx=12)

        slider_faceRecTolerance = tk.Scale(controlPanel, bg="#1E1E1E", fg="#DDDDDD",
                                           label="人脸识别容忍度：", orient=tk.HORIZONTAL, relief=tk.FLAT,
                                           bd=1, from_=0, to=1, resolution=0.01,
                                           highlightthickness=0, sliderrelief=tk.FLAT, troughcolor="gray",
                                           activebackground="#323233", width="6", sliderlength="20", length=270,
                                           variable=self.faceRecToleranceVar)
        slider_faceRecTolerance.grid(column=0, padx=12, pady=5, sticky="nw")
        tk.Label(controlPanel, text="此值越大，人脸识别的匹配就越严格。（默认：0.45）", bg="#1E1E1E", fg="#DDDDDD") \
            .grid(column=0, padx=12, pady=(0, 24), sticky="nw")

        slider_eyeClosedThreshold = tk.Scale(controlPanel, bg="#1E1E1E", fg="#DDDDDD",
                                             label="判定闭眼阈值（EAR）：", orient=tk.HORIZONTAL, relief=tk.FLAT,
                                             bd=1, from_=0, to=1, resolution=0.01,
                                             highlightthickness=0, sliderrelief=tk.FLAT, troughcolor="gray",
                                             activebackground="#323233", width="6", sliderlength="20", length=270,
                                             variable=self.eyeClosedThresholdVar)
        slider_eyeClosedThreshold.grid(column=0, padx=12, pady=5, sticky="nw")
        tk.Label(controlPanel, text="此值越大，对眨眼的判定就越宽松。（默认：0.2）", bg="#1E1E1E", fg="#DDDDDD") \
            .grid(column=0, padx=12, pady=(0, 24), sticky="nw")

        slider_mouseOpenedThreshold = tk.Scale(controlPanel, bg="#1E1E1E", fg="#DDDDDD",
                                               label="判定张嘴阈值（MAR）：", orient=tk.HORIZONTAL, relief=tk.FLAT,
                                               bd=1, from_=0, to=1, resolution=0.01,
                                               highlightthickness=0, sliderrelief=tk.FLAT, troughcolor="gray",
                                               activebackground="#323233", width="6", sliderlength="20", length=270,
                                               variable=self.mouseOpenedThresholdVar)
        slider_mouseOpenedThreshold.grid(column=0, padx=12, pady=5, sticky="nw")
        tk.Label(controlPanel, text="此值越大，对张嘴的判定就越严格。（默认：0.5）", bg="#1E1E1E",
                 fg="#DDDDDD") \
            .grid(column=0, padx=12, pady=(0, 24), sticky="nw")

        slider_expressionCheckFrequency = tk.Scale(controlPanel, bg="#1E1E1E", fg="#DDDDDD",
                                                   label="面部状态检测频率（次/每秒）：", orient=tk.HORIZONTAL,
                                                   relief=tk.FLAT,
                                                   bd=1, from_=2, to=50, resolution=1,
                                                   highlightthickness=0, sliderrelief=tk.FLAT, troughcolor="gray",
                                                   activebackground="#323233", width="6", sliderlength="20", length=270,
                                                   variable=self.expressionCheckFrequencyVar)
        slider_expressionCheckFrequency.grid(column=0, padx=12, pady=5, sticky="nw")
        tk.Label(controlPanel, text="此值越大，面部状态的检测就越精准（可能影响性能）。\n（默认：30）", bg="#1E1E1E",
                 fg="#DDDDDD") \
            .grid(column=0, padx=12, pady=(0, 24), sticky="nw")

        slider_asleepTimeThreshold = tk.Scale(controlPanel, bg="#1E1E1E", fg="#DDDDDD",
                                              label="睡觉状态判定阈值（秒）：", orient=tk.HORIZONTAL, relief=tk.FLAT,
                                              bd=1, from_=2, to=180, resolution=2,
                                              highlightthickness=0, sliderrelief=tk.FLAT, troughcolor="gray",
                                              activebackground="#323233", width="6", sliderlength="20", length=270,
                                              variable=self.asleepTimeThresholdVar)
        slider_asleepTimeThreshold.grid(column=0, padx=12, pady=5, sticky="nw")
        tk.Label(controlPanel, text="对象闭眼时间超过该值后判定为睡觉状态。（默认：60）", bg="#1E1E1E",
                 fg="#DDDDDD") \
            .grid(column=0, padx=12, pady=(0, 24), sticky="nw")

        slider_headDownAngleThreshold = tk.Scale(controlPanel, bg="#1E1E1E", fg="#DDDDDD",
                                                 label="低头判定阈值（度）：", orient=tk.HORIZONTAL, relief=tk.FLAT,
                                                 bd=1, from_=5, to=30, resolution=1,
                                                 highlightthickness=0, sliderrelief=tk.FLAT, troughcolor="gray",
                                                 activebackground="#323233", width="6", sliderlength="20", length=270,
                                                 variable=self.headDownAngleThresholdVar)
        slider_headDownAngleThreshold.grid(column=0, padx=12, pady=5, sticky="nw")
        tk.Label(controlPanel, text="对象头部俯仰角超出该值时判定为低头状态（默认：10）", bg="#1E1E1E",
                 fg="#DDDDDD") \
            .grid(column=0, padx=12, pady=(0, 24), sticky="nw")

    async def RunAsync(self):
        while True:
            self.window.update()
            await asyncio.sleep(0)  # 自动调度

    def SelectVideo(self):
        self.videoPath = TkFileDiag.askopenfilename(filetypes=[("Video", ["mp4", "flv", "avi", "rmvb"])])

    def SwitchDetectionMode(self, mode: Literal["Camera", "Video", "None"]):
        if self.task_recognition is not None:
            self.task_recognition.cancel(f"Face detection task canceled, mode switch to {mode}.")
        if self.task_updateNameBoard is not None:
            self.task_updateNameBoard.cancel("Name board update canceled.")

        # 切换模式后停止检测数据的更新，不清除列表
        for obj in self.detectionData.values():
            obj.StopOnlineDetection()

        if mode.upper() == "NONE":
            self.label_display.config(image="")
            self.hint.set("请选择文件或开启摄像头")
            return

        self.task_recognition = self.eventLoop.create_task(self.MainUpdateLoopAsync(mode))
        self.task_updateNameBoard = self.eventLoop.create_task(self.UpdateNameBoardLoopAsync())

    async def MainUpdateLoopAsync(self, mode: Literal["Camera", "Video"]):
        self.Log("Main coroutine running...")
        self.label_display.config(text="载入中...")

        # 开启视频时清除检测数据
        self.detectionData.clear()

        # 图像处理管线
        async def processPipeline(waitTime: float):
            ret, self.currentFrame = self.frameCapture.GetFrame(self.imageScaleVar.get())
            if not ret:
                return False
            self.RecognizeFaces()
            self.GetLandmarks()
            self.GetHeadPose()

            self.UpdateDetectionData()
            self.DetectExpression()

            if self.drawHeadPose.get():
                self.DrawHeadPose()
            if self.drawFaceRectVar.get():
                self.DrawDetectionData()

            self.UpdateFrame()
            await asyncio.sleep(waitTime)
            return True

        try:
            waitTime = 0.01  # 挂起时间/检测频率，注意性能影响
            if mode.upper() == "CAMERA":
                with await Frame.FromCamera() as self.frameCapture:
                    self.hint.set("正在检测摄像头中人脸")
                    self.label_display.config(text="")
                    self.Log("Detecting faces and eyes from camera...")
                    while True:
                        if not await processPipeline(waitTime):
                            break

            elif mode.upper() == "VIDEO":
                with await Frame.FromVideo(self.videoPath) as self.frameCapture:
                    self.hint.set("正在检测视频文件中人脸")
                    self.label_display.config(text="")
                    self.Log("Detecting faces and eyes from video...")
                    frameRate = 1 / self.frameCapture.GetFrameRate()
                    while True:
                        if not await processPipeline(frameRate):
                            break

            self.label_display.config(image="")
            self.hint.set("请选择文件或开启摄像头")

        except Exception:
            traceback.print_exc()
            raise

    def UpdateFrame(self):
        tkImg = Frame.BGRArray2TkImg(self.currentFrame)
        self.label_display.config(image=tkImg)
        self.label_display.image = tkImg

        t = time.time()
        self.frameRateVar.set(str(round(1 / (t - self.timestamp_display))))
        self.timestamp_display = t

    async def UpdateNameBoardLoopAsync(self):
        self.Log("Update name board coroutine running...")
        try:
            while True:
                self.UpdateNameBoard()
                await asyncio.sleep(0.2)  # FIXME: 调整以平衡性能
        except Exception:
            traceback.print_exc()
            raise

    def UpdateNameBoard(self):
        self.nameTable.delete(*self.nameTable.get_children())

        if self.faceData is not None:
            index = 1
            for info in self.detectionData.values():
                if info.ShouldDisplay:
                    value = [str(index), info.UserName, info.StatusSChinese, info.EyeBlinkTimes,
                             str(info.OnlineTime).split('.')[0]]
                    index += 1
                    self.nameTable.insert("", "end", values=value)

    def UpdateDetectionData(self):
        if self.faceData is not None:
            for name, location in self.faceData:
                if name != "Unknown":
                    if self.detectionData.get(name) is None:
                        self.detectionData[name] = DetectionInfo(name)  # 多次检查避免识别误差

                    else:
                        # 在线检查
                        self.detectionData[name].OnlineCheck()

                        # 检测对象各类阈值设定
                        self.detectionData[name].AsleepTimeThreshold = self.asleepTimeThresholdVar.get()

            nameSet = set(name for name, loc in self.faceData)  # O(n)
            offlines = filter(lambda name: name not in nameSet, self.detectionData)  # O(n)
            for name in offlines:
                self.detectionData[name].NotInCamera = True

    def RecognizeFaces(self):
        if self.currentFrame is not None:
            self.faceData = FaceRec.Recognize(self.currentFrame, FaceData.knownFaceName,
                                              FaceData.knownFaceEncoding, tolerance=self.faceRecToleranceVar.get(),
                                              model=("hog","cnn")[self.locatingModelVar.get()])
        else:
            self.Log("No frame to recognize.")

    def GetLandmarks(self):
        locations = [location for name, location in self.faceData]
        self.landmarks = FaceRec.GetLandmarks(self.currentFrame, locations)
        if self.drawLandmarkVar.get():
            FaceRec.DrawLandmarks(self.currentFrame, self.landmarks)

    def GetHeadPose(self):
        self.headPoses = []

        for landmark in self.landmarks:
            self.headPoses.append(FaceRec.GetHeadPose(landmark))

    def DetectExpression(self):  # 可转为异步降低检测频率以减少性能消耗
        if self.landmarks is not None and any(self.landmarks):
            for i in range(len(self.landmarks)):
                if self.faceData[i][0] != "Unknown":
                    info = self.detectionData[self.faceData[i][0]]
                    if FaceRec.IsEyesClosed(self.landmarks[i], self.eyeClosedThresholdVar.get()):
                        info.EyeClosed = True
                    else:
                        info.EyeClosed = False

                    if FaceRec.IsMouseOpened(self.landmarks[i], self.mouseOpenedThresholdVar.get()):
                        info.MouseOpened = True
                    else:
                        info.MouseOpened = False

                    if FaceRec.IsHeadDown(self.headPoses[i][1], self.headDownAngleThresholdVar.get()):
                        info.HeadDown = True
                    else:
                        info.HeadDown = False

    def DrawDetectionData(self):
        if self.faceData is not None:
            for name, location in self.faceData:
                FaceRec.DrawFaceRect(self.currentFrame, location)
                FaceRec.PutText(self.currentFrame, name, (location[1] + 6, location[0] + 12))
                if name != "Unknown":
                    unit = self.detectionData[name]
                    FaceRec.PutText(self.currentFrame, f"Status:{unit.Status}", (location[1] + 6, location[0] + 36))

    def DrawHeadPose(self):
        if self.headPoses is not None:
            for headpose in self.headPoses:
                FaceRec.DrawHeadPose(self.currentFrame, headpose[0])

    @staticmethod
    def Log(message):
        Debug.Log("App", message)
