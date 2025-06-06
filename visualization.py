import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_double_pendulum(t: np.ndarray,
                            Y: np.ndarray,
                            L1: float,
                            L2: float,
                            interval: int = 20,
                            save_as: str = None):
    """
    Строит анимацию движения двойного маятника.

    Параметры:
        t       (np.ndarray): одномерный массив времён (размер N).
        Y       (np.ndarray): массив состояний (N, 4) — [θ₁, θ₂, ω₁, ω₂].
        L1      (float):       длина первого маятника.
        L2      (float):       длина второго маятника.
        interval(int):         задержка между кадрами в миллисекундах (по умолчанию 20).
        save_as (str или None): если задано (например, "pendulum.mp4" или "pendulum.gif"),
                                анимация будет сохранена в указанный файл.

    Возвращает:
        ani (FuncAnimation): объект анимации. Его можно использовать для управления (pause, resume).
    """
    if Y.shape[1] < 2:
        raise ValueError("Массив Y должен иметь по крайней мере два столбца (θ₁ и θ₂).")

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-(L1 + L2) * 1.1, (L1 + L2) * 1.1),
                         ylim=(-(L1 + L2) * 1.1, (L1 + L2) * 1.1))
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        theta1_i = Y[i, 0]
        theta2_i = Y[i, 1]
        x1 = L1 * np.sin(theta1_i)
        y1 = -L1 * np.cos(theta1_i)
        x2 = x1 + L2 * np.sin(theta2_i)
        y2 = y1 - L2 * np.cos(theta2_i)
        line.set_data([0, x1, x2], [0, y1, y2])
        return (line,)

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  frames=len(t),
                                  interval=interval,
                                  blit=True,
                                  init_func=init)

    if save_as:
        # Если указано имя файла, сохраняем анимацию (mp4 или gif)
        ani.save(save_as, writer='ffmpeg', dpi=200)

    plt.show()
    return ani


def plot_theta1_vs_theta2(theta1: np.ndarray,
                          theta2: np.ndarray,
                          cmap: str = 'plasma',
                          s: float = 3,
                          save_as: str = None):
    """
    Строит фазовый портрет θ₁ vs θ₂ с автоматическим подбором масштабов по данным.

    Параметры:
        theta1 (np.ndarray): массив углов первого маятника (размер N).
        theta2 (np.ndarray): массив углов второго маятника (размер N).
        cmap   (str):        имя colormap (по умолчанию 'plasma').
        s      (float):      размер точек (по умолчанию 3).
        save_as (str или None): если задано (например, "phase_theta1_theta2.png"), график будет сохранён.
    """
    if theta1.shape != theta2.shape:
        raise ValueError("Массивы theta1 и theta2 должны иметь одинаковую форму.")

    fig, ax = plt.subplots()
    ax.set_xlabel("Угол первого маятника θ₁ (рад)")
    ax.set_ylabel("Угол второго маятника θ₂ (рад)")

    # Определяем минимальные и максимальные значения для авто-масштабирования
    min_t1, max_t1 = np.min(theta1), np.max(theta1)
    min_t2, max_t2 = np.min(theta2), np.max(theta2)
    # Добавляем небольшой отступ (5%)
    pad_t1 = (max_t1 - min_t1) * 0.05 if max_t1 != min_t1 else 0.1
    pad_t2 = (max_t2 - min_t2) * 0.05 if max_t2 != min_t2 else 0.1
    ax.set_xlim(min_t1 - pad_t1, max_t1 + pad_t1)
    ax.set_ylim(min_t2 - pad_t2, max_t2 + pad_t2)

    N = theta1.shape[0]
    scatter = ax.scatter(theta1, theta2, c=np.arange(N), cmap=cmap, s=s)
    fig.colorbar(scatter, label="Шаг интегрирования")
    plt.title("Фазовый портрет: θ₁ vs θ₂")

    if save_as:
        fig.savefig(save_as, dpi=200)

    plt.show()


def plot_phase_angles_vs_omega(theta1: np.ndarray,
                               omega1: np.ndarray,
                               theta2: np.ndarray,
                               omega2: np.ndarray,
                               save_as: str = None):
    """
    Строит два фазовых портрета: (θ₁ vs ω₁) и (θ₂ vs ω₂) в одной фигуре.

    Параметры:
        theta1 (np.ndarray): массив углов первого маятника (размер N).
        omega1 (np.ndarray): массив угловых скоростей первого маятника (размер N).
        theta2 (np.ndarray): массив углов второго маятника (размер N).
        omega2 (np.ndarray): массив угловых скоростей второго маятника (размер N).
        save_as (str или None): если задано (например, "phase_angles_vs_omega.png"), график будет сохранён.
    """
    if not (theta1.shape == omega1.shape == theta2.shape == omega2.shape):
        raise ValueError("Все входные массивы должны иметь одинаковую форму.")

    fig, ax = plt.subplots()
    ax.plot(theta1, omega1, label="Первый маятник")
    ax.plot(theta2, omega2, label="Второй маятник")
    ax.set_xlabel("Угол θ (рад)")
    ax.set_ylabel("Угловая скорость ω (рад/с)")
    ax.set_title("Фазовый портрет: θ vs ω")
    ax.legend()

    if save_as:
        fig.savefig(save_as, dpi=200)

    plt.show()


def plot_omega1_vs_omega2(omega1: np.ndarray,
                          omega2: np.ndarray,
                          save_as: str = None):
    """
    Строит фазовый портрет скоростей: ω₁ vs ω₂.

    Параметры:
        omega1 (np.ndarray): массив угловых скоростей первого маятника (размер N).
        omega2 (np.ndarray): массив угловых скоростей второго маятника (размер N).
        save_as (str или None): если задано (например, "phase_omega1_omega2.png"), график будет сохранён.
    """
    if omega1.shape != omega2.shape:
        raise ValueError("Массивы omega1 и omega2 должны иметь одинаковую форму.")

    fig, ax = plt.subplots()
    ax.plot(omega1, omega2)
    ax.set_xlabel("Угловая скорость первого маятника ω₁ (рад/с)")
    ax.set_ylabel("Угловая скорость второго маятника ω₂ (рад/с)")
    ax.set_title("Фазовый портрет: ω₁ vs ω₂")

    if save_as:
        fig.savefig(save_as, dpi=200)

    plt.show()
