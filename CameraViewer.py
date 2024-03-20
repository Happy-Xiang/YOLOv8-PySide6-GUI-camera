import cv2 # 导入 OpenCV 库
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
# 导入 PySide6 库中的用于界面开发的类和库函数
# PySide6 是一个 Qt 的 Python 绑定库，提供了一种编写 Python Qt 应用程序的方法。



class CameraViewer(QWidget): # 定义 CameraViewer 类，继承自 QWidget
    def __init__(self):
        super().__init__() # 调用父类的构造函数初始化 QWidget 中的属性
        self.capture = None # 初始化摄像头的对象为 None
        self.setup_ui() # 调用 CameraViewer 类中的方法，设置 UI 界面
        self.capture_frame = None

    def setup_ui(self): # 定义设置 UI 界面的方法
        self.setWindowTitle("Camera Viewer") # 设置窗口标题
        self.resize(640, 480) # 设置窗口大小

        self.image_label = QLabel(self) # 新建一个 QLabel 控件作为用于显示视频流的标签
        self.image_label.setMinimumSize(640, 480) # 设定标签控件的最小大小
        self.image_label.setAlignment(Qt.AlignCenter) # 设置标签控件中的内容居中对齐

        self.start_button = QPushButton("Start", self) # 新建一个 QPushButton 控件作为用于启动摄像头的按钮
        self.start_button.clicked.connect(self.start_capture) # 当按钮被点击时，调用 start_capture 函数

        main_layout = QVBoxLayout() # 新建一个 QVBoxLayout 控件作为主界面的布局管理器
        main_layout.addWidget(self.image_label) # 将标签控件添加到主界面布局管理器中
        main_layout.addWidget(self.start_button) # 将按钮控件添加到主界面布局管理器中

        self.setLayout(main_layout) # 将新建的主界面布局管理器设置为当前窗口的布局管理器

    @Slot() # 用于修饰类中的函数，使它成为信号与槽函数的连接器。这里用于修饰下面的 start_capture 函数
    def start_capture(self):
        self.capture = cv2.VideoCapture(0) # 创建一个 OpenCV 视频捕获对象

        self.timer = QTimer(self) # 创建一个 QTimer 对象
        self.timer.timeout.connect(self.display_frame) # 设置连接器为 display_frame 函数
        self.timer.start(60) # 设置一个时间间隔，60ms

    def display_frame(self): # 定义用于显示视频流的函数
        ret, self.capture_frame = self.capture.read() # 捕获摄像头的图像
        if ret: # 如果读取成功
            frame = cv2.cvtColor(self.capture_frame, cv2.COLOR_BGR2RGB) # 将图像从 BGR 格式转换为 RGB 格式
            h, w, ch = frame.shape # 获取图像的长、宽、通道数
            img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888) # 将图像数据转换为 QImage 格式
            self.image_label.setPixmap(QPixmap.fromImage(img)) # 设置标签控件中的内容为 QImage 类型

    def getcapture_frame(self):
        return self.capture_frame

if __name__ == "__main__":
    app = QApplication() # 创建一个 QApplication 对象
    viewer = CameraViewer() # 创建新的 CameraViewer 类的对象
    viewer.show() # 显示 CameraViewer 类的 GUI 界面
    app.exec_() # 开始主事件循环，等待窗口事件的发生。通过调用 sys.exit() 函数来退出该事件循环。