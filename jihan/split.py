import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv("/data/StressID/label_jihan.csv")

df["subject"] = df["subject/task"].apply(lambda x: x[:4])

# Get subject-level labels using majority class
subject_labels = df.groupby("subject")["affect3-class"].agg(lambda x: x.value_counts().idxmax()).reset_index()

# 44 (train + val) / 8 (test)
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=8, train_size=44, random_state=42)
trainval_idx, test_idx = next(sss1.split(subject_labels["subject"], subject_labels["affect3-class"]))

trainval = subject_labels.iloc[trainval_idx]
test = subject_labels.iloc[test_idx]

# 36 (train) / 8 (val)
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=8, train_size=36, random_state=42)
train_idx, val_idx = next(sss2.split(trainval["subject"], trainval["affect3-class"]))

train = trainval.iloc[train_idx]
val = trainval.iloc[val_idx]

def assign_split(subj):
    if subj in train["subject"].values:
        return "train"
    elif subj in val["subject"].values:
        return "val"
    elif subj in test["subject"].values:
        return "test"
    else:
        return "unknown"

df["split"] = df["subject"].apply(assign_split)
df.drop(columns=["subject"], inplace=True)

df.to_csv("/data/StressID/label_jihan.csv", index=False)
