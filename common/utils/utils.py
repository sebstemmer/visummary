import json


def save_json(path: str, json_for_saving: dict) -> None:
    """
    Save a JSON object to a JSON file.

    Args:
        path (str):
            Path to the new JSON file.
        json_for_saving (dict):
            JSON to be saved.

    Returns:
        None
    """
    with open(path, "w", encoding="UTF-8") as f:
        json.dump(json_for_saving, f, indent=2)


def load_json(path: str) -> dict:
    """
    Loads a JSON object from a JSON file.

    Args:
        path (str):
            Path to the JSON file.

    Returns:
        dict:
            Loaded JSON object.
    """
    with open(path, "r", encoding="UTF-8") as f:
        return json.load(f)


def save_file(path: str, content_for_saving: str) -> None:
    """
    Saves content to a file.

    Args:
        path (str):
            Path to the new file.
        content_for_saving (dict):
            Content to be saved.

    Returns:
        None
    """
    with open(path, "w", encoding="UTF-8") as f:
        f.write(content_for_saving)


def load_file(path: str) -> str:
    """
    Loads content from a file.

    Args:
        path (str):
            Path to the file.

    Returns:
        str:
            Loaded content.
    """
    with open(path, "r", encoding="UTF-8") as f:
        return f.read()
