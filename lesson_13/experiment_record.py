"""使用 JSON 保存和读取实验记录。"""

import json


def save_experiment(record: dict, path: str) -> None:
    """将实验记录写入 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_experiment(path: str) -> dict:
    """读取 JSON 文件并返回实验记录。"""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
