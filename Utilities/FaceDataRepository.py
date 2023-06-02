from __future__ import annotations

import json
import os

__all__ = ["SaveFaceData", "GetFaceData", "faceData"]

path = "Assets/faceData.json"
faceData: None | dict = None
knownFaceName: None | list = None
knownFaceEncoding: None | list = None

if not os.path.exists(path):
    with open(path, mode='w', encoding='utf-8') as ff:
        ff.write("{}")
        print("{}")

with open(path, "r") as file:
    faceData: dict = json.load(file)
    knownFaceName = list(faceData.keys())
    knownFaceEncoding = list(faceData.values())


def SaveFaceData(name: str, encodeData: list):
    faceData[name] = encodeData
    with open(path, "w") as f:
        jsonStr = json.dumps(faceData, indent=4)
        f.write(jsonStr)
    global knownFaceName, knownFaceEncoding
    knownFaceName = list(faceData.keys())
    knownFaceEncoding = list(faceData.values())


def GetFaceData(name: str):
    return faceData[name]
