#!/usr/bin/env bash
set -eo pipefail

# conda 초기화
__conda_setup="$('/home/ai/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -ne 0 ]; then
    export PATH="/home/ai/anaconda3/bin:$PATH"
else
    eval "$__conda_setup"
fi
unset __conda_setup

# donghwan 가상환경 활성화
conda activate donghwan

# 스크립트가 있는 기본 경로
BASE="/home/ai/Internship/stressID/Multimodal-Stress-state-recognition/donghwan/modeling"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Teacher 모델 학습 시작"
python "$BASE/train_teacher.py"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Student 모델 학습 시작"
python "$BASE/train_student.py"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 테스트 평가 실행"
python "$BASE/test.py"

echo "[$(date +'%Y-%m-%d %H:%M:%S')] 모든 작업 완료!"
