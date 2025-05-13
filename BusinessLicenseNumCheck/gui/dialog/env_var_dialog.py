from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from components.text_field import StyledTextField
from components.buttons import create_styled_button

import os
import json
from utils.config import load_settings, save_settings

class EnvVarDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("환경변수 설정")
        self.setMinimumSize(300, 200)

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("환경변수 값을 입력하세요:"))
        self.edit = StyledTextField()
        layout.addWidget(self.edit)

        # 기존 설정을 로드하여 텍스트 필드에 값 설정
        try:
            settings = load_settings()
            env_value = settings.get('env_var', '')
        except Exception:
            env_value = ''
        self.edit.set_value(env_value)

        btn = create_styled_button("저장", self.save_env_var)
        layout.addWidget(btn)

    def save_env_var(self):
        # 입력된 값을 읽어 설정에 저장
        value = self.edit.get_value()
        settings = load_settings()
        settings['env_var'] = value
        save_settings(settings)
        self.accept()