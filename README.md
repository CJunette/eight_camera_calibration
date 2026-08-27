# 8 路摄像头采集与标定工具

这个目录提供一个基于 OpenCV 的命令行工具，用来完成：

- 确认电脑已接入并可打开 8 个外接摄像头
- 同步预览 8 路画面
- 拍摄并保存每一次标定用棋盘格图片
- 计算每个摄像头内参
- 以指定参考摄像头为基准，计算其他摄像头相对外参

## 依赖

建议在项目已有 Python 环境中安装依赖：

```powershell
pip install -r requirements.txt
```

## 标定板参数

默认使用 `9x6` 内角点棋盘格，默认方格边长为 `0.025` 米。也支持 ChArUco：默认板为 `9x7` 个方格、方格边长 `0.025` 米、ArUco 标记边长 `0.018` 米、字典 `DICT_4X4_50`。

如果你的棋盘格不是这个规格，需要在采集时设置：

```powershell
python camera_calibration.py capture --board-size 9x6 --square-size 0.025
```

注意：`board-size` 是内角点数量，不是棋盘格方块数量。

ChArUco 采集示例（`charuco-squares` 是方格数量，标记边长必须小于方格边长）：

```powershell
python camera_calibration.py capture --board-type charuco --charuco-squares 9x7 --square-size 0.025 --marker-size 0.018 --aruco-dict DICT_4X4_50 --output runs/session_charuco
```

ChArUco 需要 OpenCV 的 ArUco 模块，因此依赖为 `opencv-contrib-python`。安装时请卸载环境中冲突的 `opencv-python` 包后再执行 `pip install -r requirements.txt`。

如果标定板由 OpenCV 4.5 或更早版本生成，且方格行数为偶数、左上角是白格，请额外加入 `--charuco-legacy-pattern`。该布局与 OpenCV 4.6 之后的默认布局不同；图案布局不匹配时，程序可能检测到 ArUco 标记却无法得到 ChArUco 角点。

## 1. 确认 8 个摄像头

```powershell
python camera_calibration.py detect --camera-count 8 --max-index 16
```

如果要确认 8 路摄像头都能以 1920x1080 打开，建议使用 MJPG 像素格式检测：

```powershell
python camera_calibration.py detect --camera-count 8 --max-index 16 --width 1920 --height 1080 --fps 30 --fourcc MJPG
```

如果输出少于 8 个 `OK`，请先检查 USB 带宽、供电、线缆、摄像头权限或是否被其他软件占用。

## 2. 拍摄标定图片

自动使用检测到的 8 个摄像头：

```powershell
python camera_calibration.py capture --samples 30 --output runs/session_001
```

指定摄像头索引：

```powershell
python camera_calibration.py capture --indices 0,1,2,3,4,5,6,7 --samples 30 --output runs/session_001
```

1920x1080 采集示例：

```powershell
python camera_calibration.py capture --indices 0,1,2,3,4,5,6,7 --width 1920 --height 1080 --fps 30 --fourcc MJPG --samples 30 --output runs/session_001
```

窗口快捷键：

- `SPACE`：当 8 个摄像头都识别到当前标定板时，保存一组图片
- `A`：开启或关闭自动采集
- `Q` 或 `ESC`：退出采集

采集结果会保存到：

- `runs/session_001/images/`：原始标定图片
- `runs/session_001/previews/`：带角点检测结果的预览图片
- `runs/session_001/manifest.json`：图片、相机索引、棋盘格参数记录

## 3. 执行内外参标定

默认以摄像头 `0` 为参考相机：

```powershell
python camera_calibration.py calibrate --output runs/session_001 --reference 0
```

ChArUco 采集的数据使用相同的标定命令：

```powershell
python camera_calibration.py calibrate --output runs/session_charuco --reference 0
```

标定结果保存到：

```text
runs/session_001/calibration/calibration_result.json
```

结果中包含：

- `intrinsics`：每个摄像头的内参矩阵、畸变系数、RMS 误差
- `extrinsics_relative_to_reference`：其他摄像头相对参考摄像头的旋转矩阵和平移向量

## 拍摄建议

- 每组图片必须让 8 个摄像头同时看到完整棋盘格，否则不会保存。
- 至少采集 8 组有效图片，建议 25 到 40 组。
- 标定板需要覆盖画面中心、四角、近距离、远距离和不同倾角。
- 8 路 USB 摄像头容易遇到带宽不足，必要时降低 `--width`、`--height` 或 `--fps`。

示例：

```powershell
python camera_calibration.py capture --indices 0,1,2,3,4,5,6,7 --width 640 --height 480 --fps 15 --samples 30 --output runs/session_001
```
