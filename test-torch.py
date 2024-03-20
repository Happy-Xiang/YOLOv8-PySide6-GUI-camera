# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 10:05:38 2022

@author: wenqingzhou@gmail.com
"""

import uuid
import cv2
import sys
from PySide6.QtCore import Qt, QSize, QTimer, QThread, Slot, Signal, QRunnable, QThreadPool, QObject
from PySide6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QMainWindow, QStatusBar, QMainWindow
from PySide6.QtGui import QPixmap, QImage, QIcon
import torch
from time import time
import numpy as np

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = torch.hub.load(r"./models/yolov8n.pt", "yolov8n", source='local', pretrained=True)
# model.to('cpu')

# img = cv2.imread(PATH_TO_IMAGE)
# classes = model.names


# results = model(imgs, size=640)  # includes NMS
def plot_boxes(results, frame):
    # labels, cord = results
    # n = len(labels)
    # x_shape, y_shape = frame.shape[1], frame.shape[0]
    # for i in range(n):
    #     row = cord[i]
    #     if row[4] >= 0.2:
    #         x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
    #         bgr = (0, 255, 0)
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
    #         # cv2.putText(frame, classes[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    #
    # return frame
    return 1


def score_frame(frame):
    """
    转换标签和坐标
    """
    # frame = [frame]
    # results = model(frame, size=640)
    # labels, cord = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
    # return labels, cord
    return 1


class Thread(QThread):
    changePixmap = Signal(QImage)

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                results = score_frame(frame)  # includes NMS
                frame = plot_boxes(results, frame)

                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)


class WorkerSignal(QObject):
    data = Signal(QImage)
    process_time = Signal(str)


class Worker(QRunnable):
    def __init__(self):
        super().__init__()
        self.job_id = uuid.uuid4().hex
        self.signal = WorkerSignal()

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                start_time = time()
                # 模型推理及绘制结果
                results = score_frame(frame)  # includes NMS
                frame = plot_boxes(results, frame)

                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                end_time = time()
                fps = 1 / np.round(end_time - start_time, 3)
                print(f"Frames Per Second : {fps:.2f}")
                self.signal.data.emit(p)
                self.signal.process_time.emit(f'{fps:.2f}')


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(r"E:\smile.ico"))
        self.initUI()

    def initUI(self):
        self.setWindowTitle('App')
        self.resize(640, 480)
        self.label = QLabel(self)
        self.label.resize(640, 480)

        self.statusbar = self.statusBar()
        # self.statusbar = QStatusBar()
        self.statusbar.showMessage('Ready')

        # QThread方法
        # self.th = Thread(self)
        # self.th.changePixmap.connect(self.setImage) # 信号与槽
        # self.th.start()

        # QThreadPool+QRunnable方法
        self.thread_pool = QThreadPool()
        self.worker = Worker()
        self.worker.signal.data.connect(self.setImage)
        self.worker.signal.process_time.connect(self.showFPS)
        self.thread_pool.start(self.worker)

        self.show()

    @Slot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    @Slot(str)
    def showFPS(self, fps):
        self.statusbar.showMessage(fps)


if __name__ == '__main__':

    # main()
    # 创建Qt应用程序
    # app = QApplication(sys.argv)
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    # app.setQuitOnLastWindowClosed(False)
    win = App()
    # win.show()
    sys.exit(app.exec())
