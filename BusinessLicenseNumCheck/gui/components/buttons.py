from PyQt5.QtWidgets import QPushButton

def create_styled_button(text, callback=None):
    button = QPushButton(text)
    button.setStyleSheet(
        "background-color: #5A6B84; color: white; border-radius: 5px; padding: 10px;"
        "font-size: 14px;"
    )
    if callback:
        button.clicked.connect(callback)
    return button
