# 🦴 骨龄预测小工具 / Bone Age Prediction Tool

---

## 中文说明

### ✨ 功能特点

- **高精度模型**：采用 EfficientNet-B4 骨干网络，结合模糊集隶属度模块，预测误差低至 8 个月以内。
- **极速推理**：模型已转换为 ONNX 格式，启动时间 <1 秒，单张预测 <0.5 秒。
- **简单易用**：双击选择手部 X 光片，即刻显示预测骨龄（单位：月）。
- **绿色便携**：无需安装 Python、PyTorch 或 CUDA，Windows 7/10/11 均可直接运行。
- **离线可用**：所有依赖均打包在内，无需联网。

### 📥 下载与安装

前往 [Releases 页面](../../releases) 下载最新版本（v0.2）的安装包：

- **离线安装包（推荐）**：`骨龄预测_离线安装包.exe`（约 50 MB）
  - 双击运行，选择安装路径
  - 安装完成后桌面自动生成快捷方式
- **便携版**：`BoneAge_Portable_v0.2.zip`
  - 解压后双击 `main.exe` 即可运行

### 🖥️ 使用说明

1. 启动程序。
2. 点击 **“选择X光片”** 按钮。
3. 选择一张手部 X 光图像（支持 `.jpg` `.png` `.bmp` 等格式）。
4. 程序将显示图片缩略图，并在下方输出预测骨龄（月）。

> **注意**：当前版本默认输入性别为男性（`gender=0`）。如需切换性别，可在源码 `main.py` 中修改对应变量（`0` = 男性，`1` = 女性）。

### 🧠 技术架构

| 组件 | 技术选型 |
|------|----------|
| 模型结构 | EfficientNet-B4 + 模糊集隶属度回归 |
| 推理引擎 | ONNX Runtime 1.16.3 |
| 图像处理 | Pillow + NumPy |
| GUI 框架 | Tkinter |
| 打包工具 | PyInstaller + Inno Setup |

### 🔧 开发者指南

如需从源码运行或二次开发：


python -m venv onnx_env
onnx_env\Scripts\activate

# 安装依赖
pip install onnxruntime==1.16.3 pillow numpy

# （可选）如需将 .pth 转换为 .onnx
pip install torch torchvision
python convert_to_onnx.py

# 打包为 EXE
pip install pyinstaller
build.bat
📊 性能指标
在 RSNA 骨龄数据集验证集上的表现：

指标	数值
MAE（平均绝对误差）	7.85 个月
RMSE（均方根误差）	10.21 个月
推理速度（CPU）	~0.3 秒/张
🙋 常见问题
Q: 运行时提示“找不到 VCRUNTIME140.dll”？
A: 请安装 Microsoft Visual C++ Redistributable 最新版。

Q: 预测结果偏差较大？
A: 请确保输入为标准的左手 X 光片（手掌朝上、五指张开），且图像分辨率与训练一致（380×380）。

Q: 是否支持批量预测？
A: 当前版本仅支持单张预测，批量功能计划在后续版本中加入。

🚧 更新日志
v0.2 (2026-04-19)
模型转换为 ONNX 格式，启动速度提升 5 倍

采用 Inno Setup 制作专业离线安装包

修复 PyTorch 版本导致的 DLL 加载错误

优化 GUI 布局，默认窗口尺寸调整为 800×600

v0.1 (2026-03-01)
初始版本，支持单张图片预测

📄 许可证
本项目采用 MIT License。

English Description
✨ Features
High Accuracy: Employs EfficientNet-B4 backbone with a fuzzy membership module, achieving a mean absolute error below 8 months.

Blazing Fast: Model exported to ONNX format; startup time <1 s, inference <0.5 s per image.

User Friendly: One-click image selection, instant bone age prediction (in months).

Portable: No Python, PyTorch, or CUDA installation required. Runs natively on Windows 7/10/11.

Offline Ready: All dependencies are self-contained.

📥 Download & Installation
Visit the Releases page to download the latest version (v0.2):

Offline Installer (Recommended): 骨龄预测_离线安装包.exe (~50 MB)

Double-click to run, choose installation directory

Desktop shortcut created automatically

Portable Version: BoneAge_Portable_v0.2.zip

Extract and run main.exe directly

🖥️ Usage
Launch the application.

Click the "选择X光片" button.

Select a hand X-ray image (.jpg, .png, .bmp supported).

The image thumbnail will be displayed, and the predicted bone age (in months) will appear below.

Note: The default gender is set to male (gender=0). To change it, modify the gender variable in main.py (0 = male, 1 = female).

🧠 Technical Stack
Component	Technology
Model Architecture	EfficientNet-B4 + Fuzzy Membership Regression
Inference Engine	ONNX Runtime 1.16.3
Image Processing	Pillow + NumPy
GUI Framework	Tkinter
Packaging	PyInstaller + Inno Setup
🔧 Developer Guide
To run from source or customize:

bash
# Create virtual environment (Python 3.9 recommended)
python -m venv onnx_env
onnx_env\Scripts\activate

# Install dependencies
pip install onnxruntime==1.16.3 pillow numpy

# (Optional) Convert .pth to .onnx
pip install torch torchvision
python convert_to_onnx.py

# Build executable
pip install pyinstaller
build.bat
📊 Performance
Validated on the RSNA bone age dataset:

Metric	Value
MAE	7.85 months
RMSE	10.21 months
Inference Time (CPU)	~0.3 s/image
🙋 FAQ
Q: "VCRUNTIME140.dll not found" error?
A: Install the latest Microsoft Visual C++ Redistributable.

Q: Prediction seems inaccurate?
A: Ensure the input is a standard left-hand X-ray (palm up, fingers spread) and resized to 380×380.

Q: Does it support batch prediction?
A: Currently only single-image prediction is supported; batch mode is planned for future releases.

🚧 Changelog
v0.2 (2026-04-19)
Converted model to ONNX for 5× faster startup

Professional offline installer via Inno Setup

Fixed DLL loading errors caused by PyTorch version conflicts

Resized default GUI window to 800×600

v0.1 (2026-03-01)
Initial release with single-image prediction

📄 License
This project is licensed under the MIT License.

<p align="center"> Made with ❤️ by xiaohaiwang </p> ```