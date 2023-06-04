from __future__ import annotations

import asyncio
from asyncio import Task
from datetime import datetime, timedelta
from typing import Any, Optional

from Final.Entities.Status import Status
from Final.Utilities import Debug


def Log(*obj: Any):
    Debug.Log("DetectionInfo", *obj)


class DetectionInfo:
    def __init__(self, userName: str):
        self.__userName: str = userName  # 用户名
        self.__onlineTime: timedelta = timedelta()  # 在线时间
        self.__status: int = Status.Init  # 当前状态
        self.__eyeBlinkTimes: int = 0  # 眨眼次数
        self.__lastTimeInCamera: datetime = datetime.now()  # 每次检测必须更新此值
        self.__notInCamera: bool = False  # 当时是否在相机/视频中
        self.__eyeClosed: bool = False  # 是否闭眼
        self.__eyeCloseTime: float = 0  # 连续闭眼的时间
        self.__asleepTimeThreshold: float = 3  # 连续闭眼被判定为睡觉的时间阈值
        self.__initCheckCount: int = 0  # 初始化检查次数
        self.__initCheckCountThreshold = 15  # 完成初始化所需要的检查次数
        self.__shouldDisplay: bool = False  # 在Init状态时为false，其余状态为true
        self.__mouseOpened: bool = False  # 是否张嘴
        self.__headDown: bool = False  # 是否低头
        self.__statusAnalysisFrequency = 20  # 状态检测频率

        self.task_stateAnalysis: Optional[Task] = None
        Log(f"Object has been created, user name: {self.__userName}")

    def __del__(self):
        self.task_stateAnalysis.cancel()

    @property
    def UserName(self) -> str:
        """
        用户名
        :return:
        """
        return self.__userName

    @property
    def OnlineTime(self) -> timedelta:
        """
        在线时间
        :return:
        """
        return self.__onlineTime

    @property
    def EyeBlinkTimes(self) -> int:
        """
        眨眼次数
        :return:
        """
        return self.__eyeBlinkTimes

    @property
    def Status(self) -> str:
        """
        当前状态，英文
        :return:
        """
        return Status.ToString(self.__status)

    @property
    def StatusSChinese(self) -> str:
        """
        当前状态，中文
        :return:
        """
        return Status.ToStringSChinese(self.__status)

    @property
    def NotInCamera(self) -> bool:
        """
        当前是否不在摄像头中
        :return:
        """
        return self.__notInCamera

    @NotInCamera.setter
    def NotInCamera(self, value: bool):
        # 如果对象当前处在初始化状态，则将初始化检查次数归零
        if self.__status == Status.Init:
            self.__initCheckCount = 0
        self.__notInCamera = value

    @property
    def EyeClosed(self) -> bool:
        return self.__eyeClosed

    @EyeClosed.setter
    def EyeClosed(self, value: bool):
        if self.__eyeClosed and not value:
            self.__eyeBlinkTimes += 1
        self.__eyeClosed = value

    @property
    def AsleepTimeThreshold(self):
        return self.__asleepTimeThreshold

    @AsleepTimeThreshold.setter
    def AsleepTimeThreshold(self, value: float):
        self.__asleepTimeThreshold = value

    @property
    def MouseOpened(self) -> bool:
        return self.__mouseOpened

    @MouseOpened.setter
    def MouseOpened(self, value: bool):
        self.__mouseOpened = value

    @property
    def HeadDown(self) -> bool:
        return self.__headDown

    @HeadDown.setter
    def HeadDown(self, value: bool):
        self.__headDown = value

    @property
    def ShouldDisplay(self) -> bool:
        """
        对象在初始化状态时为false，其余状态为true
        通过该值来决定对象是否应当显示
        :return:
        """
        return self.__shouldDisplay

    @property
    def StatusAnalysisFrequency(self):
        return self.__statusAnalysisFrequency

    @StatusAnalysisFrequency.setter
    def StatusAnalysisFrequency(self, value: int):
        self.__statusAnalysisFrequency = value

    def OnlineCheck(self):
        """
        在线检查，摄像头中检测到该对象时必须调用此方法
        :return:
        """

        # 如果对象为初始化状态，则初始化检查的次数+1。
        # 超出检查的阈值后将进入Normal状态并开始状态检测协程。
        # 如果在初始化过程中未检测出对象，则将初始化检查次数归零（见NotInCamera属性）。
        # 即连续检测到多次后该对象进入Normal状态，如果中途未检测到对象，则应当将该值置为0。
        # 对象为进入Normal状态时不应当显示在统计面板上。
        # 设立该机制可以有效避免人脸识别时偶尔出现的对象的误判。
        # 增大__initCheckCountThreshold值将增加进入Normal状态需要的连续检测次数，但会导致识别到对象时不能立刻进入到Normal状态。
        if self.__status == Status.Init:
            self.__initCheckCount += 1
            if self.__initCheckCount >= self.__initCheckCountThreshold:
                self.__status = Status.Normal
                self.__shouldDisplay = True
                self.StartStateAnalysis()

        self.__lastTimeInCamera = datetime.now()

    async def AnalyzeStateAsync(self):
        waitTime = 1 / self.__statusAnalysisFrequency

        def offlineCheck():
            # 不在镜头3秒后判定离线
            if time - self.__lastTimeInCamera > timedelta(seconds=3):
                self.__status = Status.Offline
                Log(f"{self.__userName} is offline")
            else:
                self.__onlineTime += timedelta(seconds=waitTime)  # FIXME 不精确计算，尝试用计时器取得精准时间

        async def onlineCheck():
            # 3秒内发现在线则等待3秒
            if time - self.__lastTimeInCamera < timedelta(seconds=3):
                await asyncio.sleep(3)
                # 等待3秒后仍然在线则判定在线
                if time - self.__lastTimeInCamera < timedelta(seconds=3):
                    self.__status = Status.Normal
                    Log(f"{self.__userName} is online")

        def eyeCloseCheck():
            if self.__eyeClosed:
                self.__status = Status.EyeClosed

        def eyeOpenCheck():
            if not self.__eyeClosed:
                self.__status = Status.Normal
                self.__eyeCloseTime = 0

        def mouseOpenCheck():
            if self.__mouseOpened:
                self.__status = Status.MouseOpened

        def mouseCloseCheck():
            if not self.__mouseOpened:
                self.__status = Status.Normal

        def asleepCheck():
            self.__eyeCloseTime += waitTime
            if self.__eyeCloseTime > self.__asleepTimeThreshold:
                self.__status = Status.Asleep

        def headDownCheck():
            if self.__headDown:
                self.__status = Status.HeadDown

        def headUpCheck():
            if not self.__headDown:
                self.__status = Status.Normal

        async def wakeUpCheck():
            eyeOpenTimes = 0
            while True:
                if self.__eyeClosed:
                    eyeOpenTimes = 0
                else:
                    eyeOpenTimes += 1
                    if eyeOpenTimes > 15:  # 连续睁眼次数大于15，判定醒来
                        self.__status = Status.Normal
                        break
                await asyncio.sleep(waitTime)

        while True:
            time = datetime.now()

            # 简易FSM，通过状态切换来实现较为规范的检测，缺点是同一时间只能处在一个状态
            if self.__status == Status.Normal:  # 正常状态
                offlineCheck()  # 离线检测
                mouseOpenCheck()  # 张嘴检测
                eyeCloseCheck()  # 闭眼检测
                headDownCheck()  # 低头检测

            elif self.__status == Status.EyeClosed:  # 闭眼状态
                offlineCheck()  # 离线检测
                eyeOpenCheck()  # 睁眼检测
                mouseOpenCheck()  # 张嘴检测
                asleepCheck()  # 睡觉检测

            elif self.__status == Status.MouseOpened:  # 张嘴状态
                offlineCheck()  # 离线检测
                headDownCheck()  # 低头检测
                mouseCloseCheck()  # 闭嘴检测

            elif self.__status == Status.HeadDown:  # 低头状态
                offlineCheck()  # 离线检测
                headUpCheck()  # 抬头检测

            elif self.__status == Status.Asleep:  # 睡觉状态
                await wakeUpCheck()  # 清醒检测

            elif self.__status == Status.Offline:  # 离线状态
                await onlineCheck()  # 在线检测

            await asyncio.sleep(waitTime)  # 非精确计算，受CPU和协程切换影响

    def StartStateAnalysis(self):
        """
        开始状态检查
        :return:
        """
        self.task_stateAnalysis = asyncio.create_task(self.AnalyzeStateAsync())

    def StopOnlineDetection(self):
        """
        关闭状态检查
        :return:
        """
        if self.task_stateAnalysis is not None:
            self.task_stateAnalysis.cancel()
