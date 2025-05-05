from pathlib import Path
import pandas as pd
import config
df = pd.read_csv(config.CSV_PATH)

for modality, path in {
    "video": config.VIDEO_DIR,
    "ecg": config.ECG_DIR,
    "eda": config.EDA_DIR,
    "rr": config.RR_DIR,
    "audio": config.AUDIO_DIR,
}.items():
    exists = df["subject/task"].apply(lambda name: Path(path, f"{name}.{'mp4' if modality=='video' else 'png'}").exists())
    print(f"{modality}: {exists.mean()*100:.2f}% 존재")

dist = df.groupby(["split", "affect3-class"]).size().unstack(fill_value=0)

print("Class 분포 (split별)")
print(dist)

print("샘플 수 (총합)")
print(df["split"].value_counts())