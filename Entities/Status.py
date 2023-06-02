
class Status:
    Init = 0
    Normal = 1
    EyeClosed = 2
    MouseOpened = 3
    HeadDown = 4
    Offline = 5
    Asleep = 6

    @staticmethod
    def ToString(statusEnum:int) -> str:
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
    def ToStringSChinese(statusEnum:int) -> str:
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

