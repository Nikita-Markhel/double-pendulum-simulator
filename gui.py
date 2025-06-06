import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QDoubleSpinBox, QLabel, QPushButton,
    QTabWidget, QSizePolicy
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from pendulum import DoublePendulum
from visualization import (
    plot_theta1_vs_theta2,
    plot_phase_angles_vs_omega,
    plot_omega1_vs_omega2
)


class PendulumWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Double Pendulum Simulator")
        self.resize(1000, 600)

        # Хранение данных после симуляции
        self.t = None
        self.Y = None
        self.ani = None
        self.paused = False

        # =======================
        # Виджеты ввода параметров
        # =======================
        params_widget = QWidget()
        form_layout = QFormLayout()

        # Длины L1, L2
        self.l1_spin = QDoubleSpinBox()
        self.l1_spin.setRange(0.1, 10.0)
        self.l1_spin.setSingleStep(0.1)
        self.l1_spin.setValue(2.0)
        form_layout.addRow("L₁ (м):", self.l1_spin)

        self.l2_spin = QDoubleSpinBox()
        self.l2_spin.setRange(0.1, 10.0)
        self.l2_spin.setSingleStep(0.1)
        self.l2_spin.setValue(2.0)
        form_layout.addRow("L₂ (м):", self.l2_spin)

        # Массы m1, m2
        self.m1_spin = QDoubleSpinBox()
        self.m1_spin.setRange(0.1, 10.0)
        self.m1_spin.setSingleStep(0.1)
        self.m1_spin.setValue(1.0)
        form_layout.addRow("m₁ (кг):", self.m1_spin)

        self.m2_spin = QDoubleSpinBox()
        self.m2_spin.setRange(0.1, 10.0)
        self.m2_spin.setSingleStep(0.1)
        self.m2_spin.setValue(3.0)
        form_layout.addRow("m₂ (кг):", self.m2_spin)

        # Начальные углы θ₁₀, θ₂₀ (в радианах)
        self.theta1_spin = QDoubleSpinBox()
        self.theta1_spin.setRange(-np.pi, np.pi)
        self.theta1_spin.setSingleStep(0.1)
        self.theta1_spin.setValue(np.pi / 2)
        form_layout.addRow("θ₁₀ (рад):", self.theta1_spin)

        self.theta2_spin = QDoubleSpinBox()
        self.theta2_spin.setRange(-np.pi, np.pi)
        self.theta2_spin.setSingleStep(0.1)
        self.theta2_spin.setValue(np.pi / 2)
        form_layout.addRow("θ₂₀ (рад):", self.theta2_spin)

        # Начальные угловые скорости ω₁₀, ω₂₀
        self.omega1_spin = QDoubleSpinBox()
        self.omega1_spin.setRange(-10.0, 10.0)
        self.omega1_spin.setSingleStep(0.5)
        self.omega1_spin.setValue(3.0)
        form_layout.addRow("ω₁₀ (рад/с):", self.omega1_spin)

        self.omega2_spin = QDoubleSpinBox()
        self.omega2_spin.setRange(-10.0, 10.0)
        self.omega2_spin.setSingleStep(0.5)
        self.omega2_spin.setValue(0.0)
        form_layout.addRow("ω₂₀ (рад/с):", self.omega2_spin)

        # Время моделирования t_max и шаг dt
        self.tmax_spin = QDoubleSpinBox()
        self.tmax_spin.setRange(0.1, 100.0)
        self.tmax_spin.setSingleStep(0.5)
        self.tmax_spin.setValue(25.0)
        form_layout.addRow("t_max (с):", self.tmax_spin)

        self.dt_spin = QDoubleSpinBox()
        self.dt_spin.setRange(0.001, 1.0)
        self.dt_spin.setSingleStep(0.01)
        self.dt_spin.setValue(0.03)
        form_layout.addRow("dt (с):", self.dt_spin)

        # Кнопки управления
        self.run_button = QPushButton("Запустить симуляцию")
        self.run_button.clicked.connect(self.run_simulation)
        form_layout.addRow(self.run_button)

        self.pause_button = QPushButton("Пауза")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self.toggle_pause)
        form_layout.addRow(self.pause_button)

        self.save_button = QPushButton("Сохранить данные")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_data)
        form_layout.addRow(self.save_button)

        params_widget.setLayout(form_layout)

        # ==============================
        # Настройка вкладок (Tabs)
        # ==============================
        tabs = QTabWidget()

        # --- Вкладка 1: Анимация ---
        tab_anim = QWidget()
        anim_layout = QVBoxLayout()

        # FigureCanvas для анимации
        self.canvas_anim = FigureCanvas(plt.Figure())
        self.canvas_anim.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        anim_layout.addWidget(self.canvas_anim)
        tab_anim.setLayout(anim_layout)

        # --- Вкладка 2: Статичные графики ---
        tab_plots = QWidget()
        plots_layout = QVBoxLayout()

        # Кнопки для открытия каждого статичного графика
        self.plot_theta1_theta2_btn = QPushButton("Показать фазовый портрет θ₁ vs θ₂")
        self.plot_theta1_theta2_btn.setEnabled(False)
        self.plot_theta1_theta2_btn.clicked.connect(self.show_plot_theta1_theta2)
        plots_layout.addWidget(self.plot_theta1_theta2_btn)

        self.plot_angles_vs_omega_btn = QPushButton("Показать фазовый портрет θ vs ω")
        self.plot_angles_vs_omega_btn.setEnabled(False)
        self.plot_angles_vs_omega_btn.clicked.connect(self.show_plot_angles_vs_omega)
        plots_layout.addWidget(self.plot_angles_vs_omega_btn)

        self.plot_omega1_omega2_btn = QPushButton("Показать фазовый портрет ω₁ vs ω₂")
        self.plot_omega1_omega2_btn.setEnabled(False)
        self.plot_omega1_omega2_btn.clicked.connect(self.show_plot_omega1_omega2)
        plots_layout.addWidget(self.plot_omega1_omega2_btn)

        tab_plots.setLayout(plots_layout)

        tabs.addTab(tab_anim, "Анимация")
        tabs.addTab(tab_plots, "Графики")

        # =============================
        # Основная компоновка окна
        # =============================
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.addWidget(params_widget, 1)
        main_layout.addWidget(tabs, 3)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def run_simulation(self):
        # Считываем параметры из виджетов
        L1 = float(self.l1_spin.value())
        L2 = float(self.l2_spin.value())
        m1 = float(self.m1_spin.value())
        m2 = float(self.m2_spin.value())
        theta1_0 = float(self.theta1_spin.value())
        theta2_0 = float(self.theta2_spin.value())
        omega1_0 = float(self.omega1_spin.value())
        omega2_0 = float(self.omega2_spin.value())
        t_max = float(self.tmax_spin.value())
        dt = float(self.dt_spin.value())

        # Создаём экземпляр модели и интегрируем
        pend = DoublePendulum(L1=L1, L2=L2, m1=m1, m2=m2)
        y0 = np.array([theta1_0, theta2_0, omega1_0, omega2_0], dtype=float)
        t, Y = pend.integrate(y0=y0, t_max=t_max, dt=dt, method="rk4")
        self.t = t
        self.Y = Y

        # Включаем кнопки управления анимацией и сохранением
        self.pause_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.plot_theta1_theta2_btn.setEnabled(True)
        self.plot_angles_vs_omega_btn.setEnabled(True)
        self.plot_omega1_omega2_btn.setEnabled(True)

        # Настраиваем анимацию на canvas_anim
        fig = self.canvas_anim.figure
        fig.clear()
        ax = fig.add_subplot(111, aspect='equal',
                             autoscale_on=False,
                             xlim=(-(L1 + L2) * 1.1, (L1 + L2) * 1.1),
                             ylim=(-(L1 + L2) * 1.1, (L1 + L2) * 1.1))
        ax.grid()

        line, = ax.plot([], [], 'o-', lw=2)

        def init():
            line.set_data([], [])
            return (line,)

        def animate_frame(i):
            theta1_i = Y[i, 0]
            theta2_i = Y[i, 1]
            x1 = L1 * np.sin(theta1_i)
            y1 = -L1 * np.cos(theta1_i)
            x2 = x1 + L2 * np.sin(theta2_i)
            y2 = y1 - L2 * np.cos(theta2_i)
            line.set_data([0, x1, x2], [0, y1, y2])
            return (line,)

        # Создаём FuncAnimation
        self.ani = animation.FuncAnimation(
            fig,
            animate_frame,
            frames=len(t),
            interval=int(1000 * dt),
            blit=True,
            init_func=init
        )
        self.canvas_anim.draw()

    def toggle_pause(self):
        if not self.ani:
            return
        if self.paused:
            self.ani.event_source.start()
            self.pause_button.setText("Пауза")
        else:
            self.ani.event_source.stop()
            self.pause_button.setText("Возобновить")
        self.paused = not self.paused

    def save_data(self):
        # Сохраняем данные (t и Y) в файл pendulum_data.npz
        if self.t is None or self.Y is None:
            return
        pend = DoublePendulum(
            L1=float(self.l1_spin.value()),
            L2=float(self.l2_spin.value()),
            m1=float(self.m1_spin.value()),
            m2=float(self.m2_spin.value())
        )
        pend.save_to_file(self.t, self.Y, filename="pendulum_data.npz")

    def show_plot_theta1_theta2(self):
        # Вызываем функцию из visualization для открытия нового окна с фазовым портретом θ₁ vs θ₂
        if self.Y is None:
            return
        theta1 = self.Y[:, 0]
        theta2 = self.Y[:, 1]
        plot_theta1_vs_theta2(theta1, theta2, save_as=None)

    def show_plot_angles_vs_omega(self):
        # Открываем окно с фазовым портретом θ vs ω
        if self.Y is None:
            return
        theta1 = self.Y[:, 0]
        omega1 = self.Y[:, 2]
        theta2 = self.Y[:, 1]
        omega2 = self.Y[:, 3]
        plot_phase_angles_vs_omega(theta1, omega1, theta2, omega2, save_as=None)

    def show_plot_omega1_omega2(self):
        # Открываем окно с фазовым портретом ω₁ vs ω₂
        if self.Y is None:
            return
        omega1 = self.Y[:, 2]
        omega2 = self.Y[:, 3]
        plot_omega1_vs_omega2(omega1, omega2, save_as=None)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PendulumWindow()
    window.show()
    sys.exit(app.exec_())
