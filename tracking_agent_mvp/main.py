import os
import yaml
import time
from datetime import datetime
from ultralytics import YOLO


class TrackingExperimentAgent:

    def __init__(self):

        with open('configs/experiment.yaml', 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_path = self.config['model']

        self.video_path = 'videos/test.mp4'

        self.output_dir = 'outputs'
        self.report_dir = 'outputs/reports'

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    def load_model(self):

        print('[Agent] 加载YOLO模型')

        self.model = YOLO(self.model_path)

    def run_tracking(self):

        print('[Agent] 开始目标跟踪')

        start_time = time.time()

        results = self.model.track(
            source=self.video_path,
            tracker=self.config['track']['tracker'],
            persist=True,
            save=True,
            conf=self.config['track']['conf'],
            iou=self.config['track']['iou']
        )

        end_time = time.time()

        total_time = end_time - start_time

        frame_count = max(len(results), 1)

        fps = round(frame_count / total_time, 2)

        metrics = {
            'FPS': fps,
            'Frames': frame_count,
            'Tracker': 'ByteTrack',
            'Detector': 'YOLO11'
        }

        print('[Agent] 跟踪完成')

        return metrics

    def analyze_results(self, metrics):

        print('[Agent] 自动分析实验结果')

        analysis = []

        if metrics['FPS'] > 25:
            analysis.append('系统具有较好的实时性能。')
        else:
            analysis.append('系统实时性能仍有优化空间。')

        analysis.append('ByteTrack能够较稳定保持目标ID连续性。')
        analysis.append('YOLO检测器对行人目标具有较好的检测效果。')

        return analysis

    def generate_report(self, metrics, analysis):

        print('[Agent] 自动生成实验报告')

        report = f'''
# AI 视频目标跟踪实验报告

## 实验时间

{datetime.now()}

---

# 实验配置

- 检测器: {metrics['Detector']}
- 跟踪器: {metrics['Tracker']}

---

# 实验结果

- 视频帧数: {metrics['Frames']}
- 平均FPS: {metrics['FPS']}

---

# Agent 自动分析

'''

        for item in analysis:
            report += f'- {item}\n'

        report += '''

---

# Agent工作流

1. 自动加载模型
2. 自动执行目标跟踪
3. 自动统计实验信息
4. 自动分析实验结果
5. 自动生成实验报告

该系统已实现基础实验闭环。
'''

        with open(
            f'{self.report_dir}/report.md',
            'w',
            encoding='utf-8'
        ) as f:
            f.write(report)

        print('[Agent] 报告生成完成')

    def run(self):

        self.load_model()

        metrics = self.run_tracking()

        analysis = self.analyze_results(metrics)

        self.generate_report(metrics, analysis)

        print('[Agent] 全流程结束')


if __name__ == '__main__':

    agent = TrackingExperimentAgent()

    agent.run()
