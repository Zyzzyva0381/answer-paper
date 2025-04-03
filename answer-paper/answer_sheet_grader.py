import cv2
import numpy as np
import argparse
import json
import os
from matplotlib import pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['STSong']
rcParams['axes.unicode_minus'] = False

default_params = {
    "start_x": 250.1,
    "start_y": 491.0,
    "col_gap": 1347.4,
    "row_gap": 114.4,
    "option_gap": 129.8,
    "box_w": 74.7,
    "box_h": 64.9
  }


class AnswerSheetGrader:
    def __init__(self):
        # 初始化参数
        self.answer_key = None
        self.marker_size = 40  # 定位标记大小(像素)
        self.option_count_per_question = 4
        self.threshold_fill_percentage = 0.5

        # 标准输出尺寸(A4@300dpi)
        self.std_width = 2480
        self.std_height = 3508

        # 默认检测参数(基于标准尺寸)
        self.detection_params = default_params.copy()

    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.detection_params.update(config.get('detection_params', {}))
            self.answer_key = config.get('answer_key', None)

    def set_answer_key(self, answers):
        """设置正确答案"""
        self.answer_key = [ans.upper() for ans in answers]

    def grade_answer_sheet(self, image_path, do_visualize=True, do_grading=True):
        """批改答题卡主流程"""
        # 1. 图像加载与预处理
        original_img = cv2.imread(image_path)
        if original_img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 2. 检测定位标记
        processed_img = self._preprocess_image(original_img)
        markers = self._detect_square_markers(processed_img)
        if len(markers) != 4:
            raise ValueError(f"需要4个定位标记，检测到{len(markers)}个")

        # 3. 标准化透视变换
        aligned_img = self._align_with_squares(original_img, markers)

        # 4. 答案检测与批改
        answers = self._detect_answers(aligned_img)
        results = self._grade_answers(answers, do_grading)

        # 5. 可视化结果
        if do_visualize:
            self._visualize_results(original_img, markers, aligned_img, results)
        results.update({'image_path': image_path})
        return results

    def _preprocess_image(self, img):
        """图像预处理"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _detect_square_markers(self, image):
        """检测正方形定位标记"""
        edges = cv2.Canny(image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        markers = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if (self.marker_size * 0.7 < w < self.marker_size * 1.3
                        and 0.9 < aspect_ratio < 1.1):
                    markers.append((x, y, w, h))
        return markers

    def _align_with_squares(self, original_img, markers):
        """标准化透视变换"""
        # 排序标记(左上、右上、左下、右下)
        markers = sorted(markers, key=lambda m: (m[1], m[0]))
        top_markers = sorted(markers[:2], key=lambda m: m[0])
        bottom_markers = sorted(markers[2:], key=lambda m: m[0])
        ordered_markers = top_markers + bottom_markers

        # 计算变换矩阵
        src_points = np.float32([
            [m[0] + m[2] / 2, m[1] + m[3] / 2] for m in ordered_markers
        ])
        dst_points = np.float32([
            [0, 0], [self.std_width, 0],
            [0, self.std_height], [self.std_width, self.std_height]
        ])

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(original_img, matrix,
                                   (self.std_width, self.std_height))

    def _detect_answers(self, image):
        """基于标准尺寸的答案检测"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        answers = {}
        params = self.detection_params

        for col in range(2):  # 2列
            for row in range(25):  # 每列25题
                for opt in range(4):  # 每题4选项
                    x = int(params['start_x'] + col * params['col_gap'] + opt * params['option_gap'])
                    y = int(params['start_y'] + row * params['row_gap'])
                    w, h = int(params['box_w']), int(params['box_h'])

                    roi = thresh[y:y + h, x:x + w]
                    if np.sum(roi == 255) / (w * h) > self.threshold_fill_percentage:
                        q_num = col * 25 + row + 1
                        answers.setdefault(q_num, []).append(chr(65 + opt))
        return answers

    def calibrate_positions(self, image_path):
        """增强版校准工具"""
        # 图像加载与对齐
        original_img = cv2.imread(image_path)
        processed_img = self._preprocess_image(original_img)
        markers = self._detect_square_markers(processed_img)
        aligned_img = self._align_with_squares(original_img, markers)

        # 创建校准窗口
        cv2.namedWindow('Calibration Tool', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Calibration Tool', 1000, 1400)

        # 参数调节范围
        param_limits = {
            'start_x': (0, self.std_width),
            'start_y': (0, self.std_height // 2),
            'col_gap': (self.std_width // 2, self.std_width // 1),
            'row_gap': (30, 150),
            'option_gap': (20, 200),
            'box_w': (20, 100),
            'box_h': (20, 100)
        }

        # 创建滑动条
        scale_factor = 10.0  # 小数点精度
        for param in self.detection_params:
            cv2.createTrackbar(
                f"{param}*{scale_factor}",
                'Calibration Tool',
                int(self.detection_params[param] * scale_factor),
                param_limits[param][1] * 10,
                lambda x: None
            )

        while True:
            # 更新参数值
            for param in self.detection_params:
                self.detection_params[param] = (
                        cv2.getTrackbarPos(f"{param}*{scale_factor}", 'Calibration Tool')
                        / scale_factor
                )

            # 绘制检测网格
            debug_img = aligned_img.copy()
            for col in range(2):
                for row in range(25):
                    for opt in range(4):
                        x = int(self.detection_params['start_x'] + col * self.detection_params['col_gap'] + opt *
                                self.detection_params['option_gap'])
                        y = int(self.detection_params['start_y'] + row * self.detection_params['row_gap'])
                        w = int(self.detection_params['box_w'])
                        h = int(self.detection_params['box_h'])

                        color = (0, 0, 255) if opt == 0 else (0, 255, 0)
                        cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
                        if row % 5 == 0 and opt == 0:
                            cv2.putText(debug_img, str(col * 25 + row + 1),
                                        (x - 50, y + h // 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 0, 255), 2)

            # 显示参数值
            param_text = " | ".join([f"{k}:{v:.1f}" for k, v in self.detection_params.items()])
            cv2.putText(debug_img, param_text, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow('Calibration Tool', debug_img)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("当前参数存入配置文件：")
                print(json.dumps(self.detection_params, indent=2))
                with open('config.json', 'r') as f:
                    config = json.load(f)
                config['detection_params'] = self.detection_params
                with open('config.json', 'w') as f:
                    json.dump(config, f, indent=4)
                print("配置文件保存成功！")

        cv2.destroyAllWindows()
        return self.detection_params

    def _grade_answers(self, detected_answers, do_grading):
        if not do_grading:
            results = {
                'graded': False,
                'detected_answers': detected_answers
            }
            return results
        """批改答案"""
        results = {
            'graded': True,
            'score': 0,
            'total': len(self.answer_key),
            'correct_count': 0,
            'wrong_questions': [], 
            'detected_answers': detected_answers
        }

        for q in range(1, len(self.answer_key) + 1):
            if set(detected_answers.get(q, [])) == {self.answer_key[q - 1]}:
                results['correct_count'] += 1
            else:
                results['wrong_questions'].append(q)

        results['score'] = int(100 * results['correct_count'] / results['total'])
        return results

    def _visualize_results(self, original_img, markers, aligned_img, results):
        """可视化结果"""
        # 创建结果图像
        detection_img = aligned_img.copy()
        answers = self._detect_answers(aligned_img)

        # 绘制识别结果
        for q_num, options in answers.items():
            col = 0 if q_num <= 25 else 1
            row = (q_num - 1) % 25
            for opt in options:
                x = int(self.detection_params['start_x'] +
                        col * self.detection_params['col_gap'] +
                        (ord(opt) - 65) * self.detection_params['option_gap'])
                y = int(self.detection_params['start_y'] +
                        row * self.detection_params['row_gap'])
                w = int(self.detection_params['box_w'])
                h = int(self.detection_params['box_h'])

                cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(detection_img, f"{q_num}{opt}",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 0, 0), 2)

        # 创建可视化布局
        plt.figure(figsize=(16, 12))
        if results['graded']:
            plt.suptitle(f"答题卡批改结果 - 得分: {results['score']}", fontsize=16)
        else:
            plt.suptitle("答题卡识别结果", fontsize=16)

        # 子图1：原始图像
        # plt.subplot(1, 2, 1)
        # marked_img = original_img.copy()
        # for x, y, w, h in markers:
        #     cv2.rectangle(marked_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
        # plt.imshow(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
        # plt.title('原始图像与定位标记')
        # plt.axis('off')
        
        # # 子图2：对齐图像
        # plt.subplot(2, 2, 2)
        # plt.imshow(cv2.cvtColor(aligned_img, cv2.COLOR_BGR2RGB))
        # plt.title('2. 标准化对齐图像')
        # plt.axis('off')
        
        # # 子图3：二值化图像
        # plt.subplot(2, 2, 3)
        # gray = cv2.cvtColor(aligned_img, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # plt.imshow(thresh, cmap='gray')
        # plt.title('3. 二值化处理结果')
        # plt.axis('off')

        # 子图4：识别结果
        # plt.subplot(1, 1, 1)
        plt.imshow(cv2.cvtColor(detection_img, cv2.COLOR_BGR2RGB))
        plt.title('答案识别结果')
        plt.axis('off')

        # 显示批改结果
        result_text = (
            f"总分: {results['score']} | "
            f"正确: {results['correct_count']}/{results['total']} | "
            f"错题: {', '.join(map(str, results['wrong_questions'])) or '无'}"
        ) if results['graded'] else "未批改"
        plt.figtext(0.5, 0.05, result_text, ha='center', fontsize=14)

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='标准化答题卡批改系统')
    parser.add_argument('image', help='答题卡图像路径')
    parser.add_argument("-n", "--no-grading", action="store_false", help="不进行批改")
    parser.add_argument("-p", "--para", action="store_true", help="校准参数（忽略n和v参数）")
    parser.add_argument("-v", "--visualize", action="store_true", help="可视化结果")
    args = parser.parse_args()

    grader = AnswerSheetGrader()

    if args.para:
        # 校准参数
        grader.calibrate_positions(args.image)
        return

    # 加载正确答案
    with open("config.json") as f:
        answer_file = json.load(f)['answer_key']
    if answer_file:
        with open(answer_file, "r") as f:
            answer_key = [line.strip().upper() for line in f if line.strip()]
        grader.set_answer_key(answer_key)

    # 执行批改
    if os.path.isdir(args.image):
        # 批改目录下的所有图像
        image_files = [os.path.join(args.image, f) for f in os.listdir(args.image) if f.endswith(('.png', '.jpg'))]
        results = []
        for image_file in image_files:
            try:
                result = grader.grade_answer_sheet(image_file, do_grading=args.no_grading, 
                                                   do_visualize=args.visualize)
            except Exception as e:
                results.append({
                    'image_path': image_file,
                    'error': str(e)
                })
                continue
            if result:
                results.append(result)
    else:
        # 批改单个图像
        results = grader.grade_answer_sheet(args.image, do_grading=args.no_grading, 
                                        do_visualize=args.visualize)

    # 输出结果
    if results:
        print(json.dumps(results, indent=4, ensure_ascii=False))


if __name__ == '__main__':
    main()
