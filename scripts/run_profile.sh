#!/usr/bin/env bash
# Jetson에서 실행: ONNX -> FP16 엔진 빌드 -> 지연 시간 측정 -> 레이어별 프로파일 내보내기
# 사용법: ./run_profile.sh <model.onnx> <태그> [입력shape, 예: images:1x3x640x640]
set -euo pipefail
ONNX=$1; TAG=$2; SHAPES=${3:-}
TRT=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
OUT=results/$(hostname)/$TAG
mkdir -p "$OUT"

# 1) 측정 환경 기록
{ date; uname -a; cat /etc/nv_tegra_release 2>/dev/null || true
  dpkg -l | grep -E "nvinfer|tensorrt" || true
  sudo nvpmodel -q || true; } > "$OUT/env.txt"

# 2) 엔진 빌드 (레이어 정보를 남기려면 빌드 때 detailed가 필요)
"$TRT" --onnx="$ONNX" --fp16 --saveEngine="$OUT/model.engine" \
  --profilingVerbosity=detailed ${SHAPES:+--shapes=$SHAPES} > "$OUT/build.log" 2>&1

# 3) 지연 시간 측정 (프로파일링 없이)
"$TRT" --loadEngine="$OUT/model.engine" ${SHAPES:+--shapes=$SHAPES} \
  --warmUp=5000 --iterations=300 --duration=0 --noDataTransfers > "$OUT/timing.log" 2>&1
grep -E "GPU Compute Time|Throughput" "$OUT/timing.log" | tee "$OUT/timing_summary.txt"

# 4) 레이어별 프로파일 (프로파일링은 오버헤드가 있어 3번과 따로 실행)
"$TRT" --loadEngine="$OUT/model.engine" ${SHAPES:+--shapes=$SHAPES} \
  --iterations=300 --duration=0 --dumpProfile --separateProfileRun \
  --profilingVerbosity=detailed \
  --exportProfile="$OUT/profile.json" --exportLayerInfo="$OUT/layers.json" > "$OUT/profile.log" 2>&1
echo "완료: $OUT"
