from PyQt5.QtWidgets import QLineEdit

class StyledTextField(QLineEdit):
    def __init__(self, placeholder_text=""):
        super().__init__()
        self.setPlaceholderText(placeholder_text)
        self.setStyleSheet(
            "border: 1px solid #cccccc; border-radius: 5px; padding: 5px; font-size: 14px;"
        )

    def set_value(self, value):
        self.setText(str(value))

    def get_value(self):
        return self.text()
