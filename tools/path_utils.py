import shutil
from pathlib import Path

def refresh_dir(p:Path) -> Path:
    if p.is_dir():
        msg = f"Path `{p}` already exist, If you don't want to overwrite it, stop the program (Ctrl+C), passing another directory and then run it again"
        print(msg)
        msg = "otherwise, it will just overwrite it"
        _ = input(msg)
        shutil.rmtree(p)
    p.mkdir(parents=True)
    return p
    