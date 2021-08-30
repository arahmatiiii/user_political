import json


def write_json(path: str, data: dict) -> None:
    with open(path, 'w') as outfile:
        json.dump(data, outfile, separators=(',', ':'), indent=4)

