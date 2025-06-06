# main.py

"""
Точка входа для запуска приложения Double Pendulum Simulator.

При запуске этого скрипта открывается GUI, где можно задать параметры двойного маятника,
запустить симуляцию, просмотреть анимацию и открыть фазовые портреты.
"""

import sys
from PyQt5.QtWidgets import QApplication
from gui import PendulumWindow

def main():
    app = QApplication(sys.argv)
    window = PendulumWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
