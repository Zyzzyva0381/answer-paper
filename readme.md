# Answer Sheet Grading System 答题卡批改系统

## 致谢/Acknowlegements

该项目由 [印度鹅卵石](https://space.bilibili.com/470106832) 发起并编写核心逻辑。点击超链接访问他的bilibili主页。

## 基本原理/Basic Principles
本系统通过计算机视觉技术自动识别和批改标准化答题卡。主要流程包括：  
This system uses computer vision to automatically grade standardized answer sheets. Main workflow:

1. **图像预处理**：使用OpenCV进行灰度化、二值化和降噪处理   
*Image Preprocessing*: Grayscale conversion, thresholding, and noise reduction using OpenCV

2. **定位标记识别**：检测答题卡上的4个定位标记实现图像对齐  
*Marker Detection*: Identify 4 alignment markers for sheet positioning

3. **答案检测**：通过分析填涂区域的像素密度判断选项是否被选中  
*Answer Detection*: Determine selected options by pixel density analysis 

4. **自动批改**：将检测结果与正确答案对比计算得分  
*Auto Grading*: Compare detected answers with answer key to calculate scores

## 环境配置/Environment Setup
```bash
# 安装依赖/Install dependencies
pip install -r requirements.txt
```
注意使用 Python 3.11.9 以防止兼容性问题

## 文件结构/File Structure
```
answer-paper/
├── answers.txt          # 正确答案文件/Answer key
├── config.json          # 检测参数配置/Detection parameters
├── answer_sheet_grader.py # 核心处理代码/Core processing code
└── requirements.txt     # 依赖列表/Dependencies
```

## 使用方法/Usage

### 基本使用/Basic Usage

```bash
python answer_sheet_grader.py --help
```

查看帮助。

```bash
python answer_sheet_grader.py image_path_or_dir [-n] [-v]
```
可以指定文件名或目录。

```bash
python answer_sheet_grader.py image_path_or_dir [-n] [-v] > results.json
```

可以将结果保存在JSON文件中。

### 校准/Calibration

```bash
python answer_sheet_grader.py image_path_or_dir -p
```

有可视化的校准参数界面，极其方便地校准参数。

### 高级配置/Advanced Configuration
也可以修改config.json手动调整检测参数：  
Modify config.json to adjust detection parameters:

```json
{
    "detection_params": {
        "start_x": 264.8,    # 起始X坐标
        "start_y": 491.0,    # 起始Y坐标
        "col_gap": 1344.4,   # 列间距
        "row_gap": 113.7,    # 行间距
        "option_gap": 112.0, # 选项间距
        "box_w": 58.8,       # 选项框宽度
        "box_h": 64.9        # 选项框高度
    },
    "answer_key": "answers.txt"
}
```

## 支持格式/Supported Formats
- 图片格式：PNG, JPG
- 答案文件：每行一个答案（A/B/C/D）
