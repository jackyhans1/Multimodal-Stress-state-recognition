#────────────────────────────────── 모두 학습+평가 ──────────────────────────────
# (가상환경 = donghwan, 루트 = modelling2 디렉터리 기준)


cd /home/ai/Internship/stressID/Multimodal-Stress-state-recognition/donghwan/modeling2

for MOD in video ecg eda rr; do
  echo "===== [${MOD^^}] Teacher 학습 시작 ====="
  python train_teacher.py  --modal $MOD       || exit 1

  echo "===== [${MOD^^}] Student 학습 시작 ====="
  python train_student.py --modal $MOD       || exit 1

  echo "===== [${MOD^^}] Student 테스트 ====="
  python test_student.py  --modal $MOD       || exit 1
done

echo "===== [ENSEMBLE] 4-Student 평균 테스트 ====="
python test_student.py --modal ensemble      || exit 1
#───────────────────────────────────────────────────────────────────────────────
