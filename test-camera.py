
import sys
import time

import cv2
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


# 定义一个QWidget类型的窗口类
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口标题和大小
        self.setWindowTitle("利用OpenCV调用摄像头")
        self.setGeometry(100, 100, 640, 480)

        # 添加按钮
        # button = QPushButton('Click me', QWidget)
        # button.resize(100, 40)
        # button.move(50, 50)

        # 创建一个QLabel控件用于显示视频流
        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 640, 480)

        # 创建视频线程并开启
        # self.thread = self.start_camera()




    # 用于调用摄像头并显示视频流的函数
    def start_camera(self):
        # 打开电脑摄像头获取视频流
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 打开默认摄像头

        # 设置视频捕获的分辨率
        # 建议选择较低分辨率
        # 因为较高分辨率的视频可能导致卡顿，影响性能
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 判断摄像头是否正常开启
        if not self.cap.isOpened():
            # 如果摄像头没有正常开启，则弹出一个警告框
            QMessageBox.warning(self, "警告", "摄像头无法打开")

            # 退出程序
            sys.exit()

        # 不断循环获取视频帧并显示
        while True:
            # 从摄像头获取一帧视频
            success, frame = self.cap.read()

            # 判断获取视频是否成功
            if success:
                print(11)
                # 将获取到的视频帧转化为Qt中的QImage类
                image = QImage(frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888)

                # 将QImage类的图像显示在QLabel控件中
                self.label.setPixmap(QPixmap.fromImage(image))

                # 设置等待时间，即视频帧之间的间隔时间
                # 建议选择合适的时间，因为间隔时间过短也可能导致卡顿
                # cv2.waitKey(1000)
                time.sleep(1)
            else:
                # 如果获取视频失败，则退出循环
                break

        # 释放摄像头资源并关闭窗口
        self.cap.release()
        self.close()


# 创建PySide6应用程序
app = QApplication(sys.argv)

# 创建自定义窗口类的实例
window = MainWindow()

# 显示窗口
window.show()
time.sleep(3)
window.start_camera()

# 运行应用程序
sys.exit(app.exec())