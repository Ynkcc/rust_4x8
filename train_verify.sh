#!/bin/bash
#
# train_verify.sh — 快速基线训练验证编排脚本
#
# 流程：
#   1. 清理旧的基线产物（日志 / 基线模型 / 指标文件）
#   2. 运行 run_baseline.py（超时保护，真实闭环短跑）
#   3. 打印日志关键段
#   4. 运行 validate_baseline.py 断言训练方向是否正确
#   5. 按校验结果返回退出码（0=通过可长跑，非 0=需检查训练逻辑）
#
# 前置：需先 `maturin develop --features pyo3` 提供 banqi_4x8 绑定。
#
set -u

cd "c:/Users/65350/Desktop/banqi/rust_4x8" || exit 1

PY="c:/Users/65350/miniconda3/python.exe"

# 基线运行超时（秒）：略大于 run_baseline.py 的 MAX_SECONDS 以留出优雅退出余量
TIMEOUT=420

# ---- 1. 清理旧产物 ----
echo "=== [1/4] 清理旧基线产物 ==="
rm -f train_verify.log train_verify_err.log
rm -f train_baseline_metrics.json
rm -f banqi_model_baseline.pt banqi_model_baseline.pth
echo "已清理日志 / 指标 / 基线模型"

# ---- 2. 运行基线训练 ----
echo ""
echo "=== [2/4] 运行基线训练 (timeout=${TIMEOUT}s) ==="
timeout "$TIMEOUT" "$PY" -u python/run_baseline.py > train_verify.log 2> train_verify_err.log
CODE=$?
echo "exit_code=$CODE"

# ---- 3. 打印日志关键段 ----
echo ""
echo "=== [3/4] STDOUT 关键段 ==="
if [ -s train_verify.log ]; then
    grep -E "\[Baseline\]|\[Training\]|\[Worker-|\[Main\]|\[Archiver\]" train_verify.log | tail -n 60
else
    echo "(stdout 为空)"
fi

if [ -s train_verify_err.log ]; then
    echo ""
    echo "=== STDERR ==="
    tail -n 40 train_verify_err.log
fi

# ---- 4. 校验指标 ----
echo ""
echo "=== [4/4] 校验基线指标 ==="
if [ "$CODE" -eq 124 ]; then
    echo "[train_verify] ⚠️ 基线运行超时被 timeout 终止（exit=124），仍尝试校验已落盘指标"
fi

"$PY" python/validate/validate_baseline.py train_baseline_metrics.json
VCODE=$?

echo ""
echo "=== 结果 ==="
echo "run_baseline exit=$CODE, validate exit=$VCODE"
if [ "$VCODE" -eq 0 ]; then
    echo "✅ 基线验证通过：训练走在正确道路上，可继续长时间运行"
    exit 0
else
    echo "❌ 基线验证未通过：请检查训练逻辑（详见上方 FAIL 项）"
    exit "$VCODE"
fi
