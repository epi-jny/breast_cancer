import json
import os
import re

path = os.path.expanduser("~/.kaggle/kaggle.json")
raw = open(path, "rb").read()
print("bytes        :", len(raw))
print("has CR (\\r)  :", b"\r" in raw)
print("has trailing newline:", raw.endswith(b"\n"))

data = json.loads(raw)
print("json keys    :", sorted(data.keys()))

username = data.get("username", "")
key = data.get("key", "")
print("username     :", repr(username))
print("key length   :", len(key))
print("key is 32-hex:", bool(re.fullmatch(r"[0-9a-f]{32}", key)))
print("key stripped == key:", key == key.strip())
