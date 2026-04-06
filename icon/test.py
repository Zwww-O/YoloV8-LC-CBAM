import sys
from PyQt5.QtWidgets import QApplication, QLabel, QWidget

app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("PyQt5 Test")
label = QLabel("PyQt5 安装成功！", window)
window.resize(300, 100)
window.show()
sys.exit(app.exec_())
