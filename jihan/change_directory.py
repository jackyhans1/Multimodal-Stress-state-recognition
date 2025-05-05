import pathlib
import shutil

SRC_DIR = pathlib.Path("/data/StressID/Videos")

DST_DIR = pathlib.Path("/data/StressID/jihan/video")

DST_DIR.mkdir(parents=True, exist_ok=True)

for wav_file in SRC_DIR.rglob("*.mp4"):
    dst_file = DST_DIR / wav_file.name
    shutil.copy2(wav_file, dst_file)
