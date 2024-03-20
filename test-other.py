import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt6.QtCore import QTimer, Qt
import cv2
from PySide6.QtGui import QImage, QPixmap


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 创建一个窗口
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle("显示视频流")

        # 创建一个文字标签并显示提示信息
        self.label = QLabel(self)
        self.label.setText("请点击下面的按钮开始录制视频：")
        self.label.setGeometry(25, 25, 700, 50)

        # 创建一个按钮并关联槽函数
        self.button = QPushButton(self)
        self.button.setText("开始录制")
        self.button.setGeometry(300, 150, 200, 100)
        self.button.clicked.connect(self.startVideo)

    def startVideo(self):
        self.label.setText("正在录制视频...")

        # 打开摄像头
        cap = cv2.VideoCapture(0)
        # 设置视频参数，宽度为800，高度为600
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

        # 创建计时器对象，定时刷新视频画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.displayVideoFrame)
        self.timer.start(50)  # 刷新间隔：50ms

    def displayVideoFrame(self):
        # 读取视频帧
        ret, frame = cap.read()
        if ret:
            # 将OpenCV的BGR格式图像转换为QImage格式
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            qimg = QImage(img.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            # 显示图像
            self.label.setPixmap(pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
