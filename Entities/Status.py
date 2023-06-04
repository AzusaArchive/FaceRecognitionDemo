class Status:
    Init = 0  # 初始化，处在该状态不应当被显示在线
    Normal = 1  # 正常
    EyeClosed = 2  # 闭眼
    MouseOpened = 3  # 张嘴
    HeadDown = 4  # 低头
    Offline = 5  # 离线
    Asleep = 6  # 睡觉

    @staticmethod
    def ToString(statusEnum: int) -> str:
        if statusEnum == 0:
            return ""
        elif statusEnum == 1:
            return "Normal"
        elif statusEnum == 2:
            return "Eye Closed"
        elif statusEnum == 3:
            return "Mouse Opened"
        elif statusEnum == 4:
            return "Head Down"
        elif statusEnum == 5:
            return "Offline"
        elif statusEnum == 6:
            return "Asleep"
        else:
            raise "查无此状态"

    @staticmethod
    def ToStringSChinese(statusEnum: int) -> str:
        if statusEnum == 0:
            return ""
        elif statusEnum == 1:
            return "正常"
        elif statusEnum == 2:
            return "闭眼"
        elif statusEnum == 3:
            return "张嘴"
        elif statusEnum == 4:
            return "低头"
        elif statusEnum == 5:
            return "离线"
        elif statusEnum == 6:
            return "睡觉"
        else:
            raise "查无此状态"
