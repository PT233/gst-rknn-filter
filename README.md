# GStreamer RKNN 插件

在 Rockchip NPU（RK3588、RK3576、RK3568 等）上运行视觉模型（YOLO、RetinaFace、PPOCR 等）的 GStreamer 插件。支持实时视频推理，可与 `videotestsrc`、`v4l2src`、`uridecodebin` 等源组合使用。

只支持model文件夹下罗列出来的模型，模型全部出自于https://github.com/airockchip/rknn_model_zoo 下转出的rknn模型。

[免责申明]该插件属于自己自娱自乐，大部分内容参考于https://github.com/haydenee/gstreamer-rknn

---

## 快速安装

### 1. 环境要求

| 项目 | 说明 |
|------|------|
| 平台 | RK3588 / RK3576 / RK3568 等（arm64） |
| 系统 | 板端 Linux（非交叉编译时） |
| 工具 | meson、ninja、pkg-config |
| 运行时 | GStreamer 1.18+、librga、librknnrt |

### 2. 安装依赖

```bash
# Debian/Ubuntu 系
sudo apt update
sudo apt install -y meson ninja-build pkg-config \
  gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

**RKNN / RGA 库**：优先使用系统安装版本；若无，则使用项目自带预编译库。

```bash
# 若系统有 librknnrt、librga（如官方镜像已集成），无需额外操作
# 否则，按 thirdparty/README.md 放入预编译库：
#   - librga: thirdparty/librga/libs/
#   - librknnrt: thirdparty/librknn_api/libs/
```

### 3. 编译与测试

```bash
cd rknnfilter
./build.sh
```

编译成功后，插件在 `build/src/libgstrknn.so`。运行测试前设置插件路径：

```bash
export GST_PLUGIN_PATH="$(pwd)/build/src:$GST_PLUGIN_PATH"
./run_test.sh
```

若 `model/` 下没有 `yolov5.rknn` 或 `yolov5s-640-640.rknn`，先放入一个 YOLOv5 的 RKNN 模型，再执行 `./run_test.sh`。

---

## 详细安装步骤



### 一、系统依赖

| 依赖 | 用途 |
|------|------|
| meson | 构建系统 |
| ninja | 编译器 |
| gstreamer-1.0 | GStreamer 核心 |
| librga | Rockchip RGA 加速 |
| librknnrt | RKNN 运行时 |

若板端为 Rockchip 官方系统，`librga` 与 `librknnrt` 通常已预装；否则需从以下地址获取预编译库：

- librga: https://github.com/airockchip/librga  
- librknnrt: https://github.com/airockchip/rknn-toolkit2 → `rknpu2/runtime/`  

将对应架构的 `librga.so`、`librknnrt.so` 放入 `thirdparty/librga/libs/` 和 `thirdparty/librknn_api/libs/`。

### 二、编译

```bash
./build.sh
# 或手动：meson setup build && ninja -C build
```

构建产物：`build/src/libgstrknn.so`。

### 三、配置插件路径

GStreamer 需能找到插件，任选其一：

**方式 A：每次运行前设置环境变量（推荐）**

```bash
export GST_PLUGIN_PATH="$(pwd)/build/src:$GST_PLUGIN_PATH"
```

**方式 B：安装到系统**

```bash
sudo ninja -C build install
# 若安装到 /usr/local，需将 /usr/local/lib/gstreamer-1.0 加入 GST_PLUGIN_PATH
```

### 四、模型准备

1. 将 RKNN 模型放入 `model/` 目录，例如 `yolov5.rknn`。
2. 检测类模型需提供 `model-type` 和标签文件（如 `coco_80_labels_list.txt`）。

可从 [RKNN Model Zoo](https://github.com/airockchip/rknn-model-zoo) 或官方网盘下载并转换模型；本仓库 `model/` 目录已包含 `coco_80_labels_list.txt`。

### 五、运行测试

```bash
# 单模型测试（YOLOv5）
./run_test.sh

# 批量测试 model/ 下所有模型（快速模式：每模型 30 帧）
./run_batch_test.sh

# 深度测试（每模型 300 帧 × 2 轮）
./run_batch_test.sh deep
```

---

## 手动运行示例（绝对路径）

```bash
gst-launch-1.0 videotestsrc num-buffers=300 ! \
  video/x-raw,format=NV12,width=640,height=480 ! \
  rknnfilter model-path=/path/to/model/yolov5.rknn model-type=yolov5 \
    label-path=/path/to/model/coco_80_labels_list.txt show-fps=true ! \
  videoconvert ! autovideosink
```

---

## 支持的模型类型（model-type）
拉到最后查看属性
```bash
gst-inspect-1.0 rknnfilter
```

| model-type | 说明 |
|------------|------|
| yolov5, yolov6, yolov7, yolov8 | 目标检测 |
| yolov8_obb | 旋转框检测 |
| yolov10, yolo11, ppyoloe | 目标检测 |
| yolov8_pose | 人体关键点 |
| yolov5_seg, yolov8_seg, ppseg | 图像分割 |
| deeplabv3 | 语义分割 |
| retinaface | 人脸检测 |
| lprnet | 车牌识别 |
| ppocr_det, ppocr_rec | 文字检测/识别 |

---

## 常见问题

### 1. 找不到插件 `rknnfilter`

```bash
export GST_PLUGIN_PATH="$(pwd)/gstreamer-rknn/build/src:$GST_PLUGIN_PATH"
# 或在 gstreamer-rknn 目录下执行
```

### 2. 提示找不到 librknnrt 或 librga

- 确认系统或 `thirdparty/*/libs/` 中有对应架构的 `.so` 文件。
- 运行时将包含这些库的目录加入 `LD_LIBRARY_PATH`。

### 3. meson 报错找不到 gstreamer

```bash
sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

### 4. run_test.sh 报错找不到模型

将 `yolov5.rknn` 或 `yolov5s-640-640.rknn` 放到 `gstreamer-rknn/model/` 目录。

### 5. OpenCV 可选

meson 会检测 `opencv4`；未安装时仍可编译，部分后处理功能可能受限。

---

## 相关链接

- [gstreamer rknn](https://github.com/haydenee/gstreamer-rknn)
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2)  
- [RKNN Model Zoo](https://github.com/airockchip/rknn-model-zoo)  
- [librga](https://github.com/airockchip/librga)  

---

