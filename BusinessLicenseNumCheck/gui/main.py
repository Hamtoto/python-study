from PyQt5.QtWidgets import QApplication
from frame import MainFrame

if __name__ == "__main__":
    app = QApplication([])
    window = MainFrame()
    window.show()
    app.exec_()