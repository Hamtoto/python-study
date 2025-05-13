import os
import sys
from PyQt5.QtWidgets import QMainWindow, QStackedWidget, QVBoxLayout, QWidget, QHBoxLayout, QLabel, QPushButton, QAction, QMessageBox
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QIcon

from pages.excel_converter import ExcelConverterPage
from dialog.env_var_dialog import EnvVarDialog

class MainFrame(QMainWindow):
    def __init__(self):
        super().__init__()
        # 프로그램 제목 및 창 설정
        self.setWindowTitle("BLNC")
        self.setGeometry(100, 100, 600, 800)
        self.setMinimumSize(QSize(600, 800))

        # 아이콘 설정
        icon_path = self.get_icon_path()
        self.setWindowIcon(QIcon(icon_path))
            
        # Base directory 설정
        self.base_dir = os.path.dirname(__file__)

        # 사이드바 메뉴 구성
        self.menu_config = {
            "header_title": "BLNC"
        }
        self.initUI()

    def get_icon_path(self):
        # OS에 따라 아이콘 경로 설정
        base_dir = os.path.dirname(__file__)
        if sys.platform.startswith("win"):
            return os.path.join(base_dir, "src", "img", "logo.ico")
        elif sys.platform.startswith("darwin"):
            return os.path.join(base_dir, "src", "img", "logo.png")
        else:
            return os.path.join(base_dir, "src", "img", "logo.png")

    def initUI(self):
        # 메인 위젯 설정
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #f5f6fa;")
        self.setCentralWidget(main_widget)

        # 레이아웃 설정
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.pages = QStackedWidget()

        # 페이지 생성

        self.excel_converter_page = ExcelConverterPage()
        
        self.pages.addWidget(self.excel_converter_page)

        # 사이드바 생성
        # sidebar = self.create_sidebar()

        # 레이아웃 구성
        # main_layout.addWidget(sidebar)
        main_layout.addWidget(self.pages)
        main_widget.setLayout(main_layout)

        # 상단 시스템 프레임 유지
        self.setWindowFlags(self.windowFlags() & ~Qt.FramelessWindowHint)
        # 기본 메뉴바 생성
        menubar = self.menuBar()
        # macOS에서도 윈도우 안에 메뉴바를 보이게 설정
        menubar.setNativeMenuBar(False)
        # 설정 메뉴 추가
        fileMenu = menubar.addMenu("설정")

        envAction = QAction("환경변수 설정", self)
        aboutAction = QAction("About", self)

        envAction.triggered.connect(lambda: EnvVarDialog(self).exec_())
        aboutAction.triggered.connect(lambda: QMessageBox.information(self, "About", "BLNC v1.0"))

        fileMenu.addAction(envAction)
        fileMenu.addAction(aboutAction)
        # 전체 위젯 기본 스타일 설정
        self.setStyleSheet("""
        QLineEdit {
            border: 1px solid #cccccc;
            border-radius: 5px;
            padding: 5px;
            font-size: 14px;
        }
        QPushButton {
            background-color: #5A6B84;
            color: white;
            border-radius: 5px;
            padding: 10px;
            font-size: 14px;
        }
        """)
        self.show()

    #이미지 로드
    def load_image(self, relative_path, size=None):
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(self.base_dir)
        absolute_path = os.path.join(base_path, relative_path)
        if not os.path.exists(absolute_path):
            print(f"Error: Image file not found at {absolute_path}")
            return QPixmap()  # 빈 QPixmap 반환
        pixmap = QPixmap(absolute_path)
        if size:
            pixmap = pixmap.scaled(size[0], size[1], Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pixmap
    
    #사이드바 관련 코드 
    """
    def create_sidebar(self):
        # 사이드바 생성
        sidebar = QWidget()
        layout = QVBoxLayout()

        # PNG 아이콘 경로 설정
        logo_pixmap = self.load_image("src/img/logo.png", size=(40, 40))
        sidebar.setFixedWidth(250)
        sidebar.setStyleSheet("background-color: #262C39; color: white;")

        # 상단 로고 및 제목
        header_layout = QHBoxLayout()
        logo_label = QLabel()
        logo_label.setPixmap(logo_pixmap)
        title_label = QLabel(self.menu_config.get("header_title", "Not Configured"))
        title_label.setStyleSheet("font-size: 20px; color: white; font-weight: bold; margin-left: 10px;")
        header_layout.addWidget(logo_label)
        header_layout.addWidget(title_label)
        header_layout.setAlignment(Qt.AlignCenter)
        layout.addLayout(header_layout)

        # --- WineCuration 메뉴 섹션 ---
        wine_section_label = QLabel(self.menu_config.get("subtitle_1", "Not Configured"))
        wine_section_label.setStyleSheet("font-size: 14px; color: #B6BDCE; margin: 10px;")
        layout.addWidget(wine_section_label)

        wine_menu_items = self.menu_config.get("winecuration_menu", [])
        for item in wine_menu_items:
            page_name = item.get("name", "")
            if page_name == "Image Downloader":
                callback = lambda: self.pages.setCurrentWidget(self.image_download_page)
            elif page_name == "Image Processing":
                callback = lambda: self.pages.setCurrentWidget(self.image_processing_page)
            elif page_name == "File Check":
                callback = lambda: self.pages.setCurrentWidget(self.file_check_page)
            else:
                callback = lambda: print(f"Unhandled WineCuration page name: {page_name}")
            menu_button = self.create_sidebar_button(
                text=page_name,
                callback=callback,
                icon_path=os.path.join(self.base_dir, item.get("icon", ""))
            )
            layout.addWidget(menu_button)
        
        layout.addStretch()

        sidebar.setLayout(layout)
        return sidebar

    
    def create_sidebar_button(self, text, callback, icon_path=None):
        button = QPushButton(text)
        button.setStyleSheet(
            "color: white; border-radius: 5px; padding: 10px;"
            "font-size: 14px; margin: 5px; text-align: left;"
        )
        button.clicked.connect(callback)
        # PNG 아이콘 설정
        if icon_path:
            print(f"Loading icon from: {icon_path}")
            if os.path.exists(icon_path):
                button.setIcon(QIcon(icon_path))
                button.setIconSize(QSize(20, 20))
            else:
                print(f"Error: Icon file not found at {icon_path}")
        return button    
    """