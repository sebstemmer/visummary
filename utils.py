import json


def save_json(path: str, json_for_saving: dict) -> None:
    with open(path, "w", encoding="UTF-8") as f:
        json.dump(json_for_saving, f, indent=2)
