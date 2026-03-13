#!/bin/bash
# 批量测试 gstreamer-rknn/model 下 RKNN 模型
# 用法: ./run_batch_test.sh          # 快速测试 (30帧)
#       ./run_batch_test.sh deep     # 深度测试 (300帧, 2轮, 更长超时)
# 使用 fakesink 无头运行，无需显示器
# 注: YOLOX 已移除支持

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model"
LABEL_PATH="${MODEL_DIR}/coco_80_labels_list.txt"
GST_LAUNCH="gst-launch-1.0"
export GST_DEBUG=0

# 深度测试模式
if [ "$1" = "deep" ]; then
    NUM_BUFFERS=300
    TIMEOUT_SEC=180
    NUM_ROUNDS=2
    echo "【深度测试】每模型 ${NUM_BUFFERS} 帧 x ${NUM_ROUNDS} 轮, 超时 ${TIMEOUT_SEC}s"
else
    NUM_BUFFERS=30
    TIMEOUT_SEC=90
    NUM_ROUNDS=1
fi

# 可选跳过列表，如: SKIP_MODELS="xxx.rknn"
SKIP_MODELS="${SKIP_MODELS:-}"

run_cmd() { if command -v timeout >/dev/null 2>&1; then timeout $TIMEOUT_SEC "$@"; else "$@"; fi; }

# 模型文件名 -> model-type 映射 (与 rknnprocess.h/gstrknn.c 保持一致)
get_model_type() {
    case "$1" in
        yolov5.rknn|yolov5s-640-640.rknn) echo "yolov5";;
        yolov6.rknn|yolov6*.rknn) echo "yolov6";;
        yolov7.rknn) echo "yolov7";;
        yolov8.rknn) echo "yolov8";;
        yolov8_obb.rknn) echo "yolov8_obb";;
        yolov10.rknn) echo "yolov10";;
        yolo11.rknn) echo "yolo11";;
        ppyoloe.rknn) echo "ppyoloe";;
        yolov8_pose.rknn) echo "yolov8_pose";;
        yolov5_seg.rknn) echo "yolov5_seg";;
        yolov8_seg.rknn) echo "yolov8_seg";;
        ppseg.rknn) echo "ppseg";;
        deeplabv3.rknn) echo "deeplabv3";;
        RetinaFace.rknn) echo "retinaface";;
        LPRNet.rknn) echo "lprnet";;
        PPOCR-Det.rknn) echo "ppocr_det";;
        PPOCR-Rec.rknn) echo "ppocr_rec";;
        yolox.rknn|yolox*.rknn) echo "";;  # YOLOX 已移除支持，返回空以跳过
        *) echo "";;
    esac
}

if [ ! -f "$LABEL_PATH" ]; then
    echo "Warning: labels not found at $LABEL_PATH (optional for some models)"
fi

echo "=========================================="
echo "批量测试 RKNN 模型 (model dir: $MODEL_DIR)"
echo "每个模型 ${NUM_BUFFERS} 帧 x ${NUM_ROUNDS} 轮"
echo "=========================================="

PASS=0
FAIL=0
SKIP=0

for model in "$MODEL_DIR"/*.rknn; do
    [ -f "$model" ] || continue
    base=$(basename "$model")
    mtype=$(get_model_type "$base")

    if [ -z "$mtype" ]; then
        if [ "$base" = "yolox.rknn" ] || echo "$base" | grep -q "^yolox"; then
            echo "[跳过] $base - YOLOX 已移除支持"
        else
            echo "[跳过] $base - 未识别的模型类型"
        fi
        SKIP=$((SKIP + 1))
        continue
    fi
    if [ -n "$SKIP_MODELS" ] && echo ",$SKIP_MODELS," | grep -q ",$base," 2>/dev/null; then
        echo "[跳过] $base - 在跳过列表中 (SKIP_MODELS)"
        SKIP=$((SKIP + 1))
        continue
    fi

    all_ok=1
    for round in $(seq 1 $NUM_ROUNDS); do
        if [ $NUM_ROUNDS -gt 1 ]; then
            printf "[测试] %-20s (model-type=%s) 第%d/%d轮 ... " "$base" "$mtype" "$round" "$NUM_ROUNDS"
        else
            printf "[测试] %-20s (model-type=%s) ... " "$base" "$mtype"
        fi

        run_cmd $GST_LAUNCH videotestsrc num-buffers=$NUM_BUFFERS ! \
            video/x-raw,format=NV12,width=640,height=480 ! \
            rknnfilter model-path="$model" model-type="$mtype" \
                label-path="$LABEL_PATH" show-fps=true silent=true ! \
            fakesink sync=true >/dev/null 2>&1
        ret=$?

        if [ $ret -eq 0 ]; then
            echo "通过"
        elif [ $ret -eq 124 ]; then
            echo "超时"
            all_ok=0
            break
        else
            echo "失败 (exit=$ret)"
            all_ok=0
            break
        fi
    done
    if [ $all_ok -eq 1 ]; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=========================================="
echo "结果: 通过=$PASS 失败=$FAIL 跳过=$SKIP"
echo "=========================================="

[ $FAIL -gt 0 ] && exit 1
exit 0
