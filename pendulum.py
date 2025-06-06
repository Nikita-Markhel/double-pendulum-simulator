
"""
Модуль с классом DoublePendulum для расчёта динамики двойного маятника.

Класс реализует:
- вычисление производных (θ₁, θ₂, ω₁, ω₂) по формулам Лагранжа;
- интегрирование методом Рунге–Кутты 4-го порядка (RK4);
- опционально: сохранение результатов в файл.
"""

import numpy as np


class DoublePendulum:
    """
    Класс модели двойного маятника.

    Атрибуты:
        L1 (float): длина первого маятника (м).
        L2 (float): длина второго маятника (м).
        m1 (float): масса первого маятника (кг).
        m2 (float): масса второго маятника (кг).
        g  (float): ускорение свободного падения (м/с²), по умолчанию 9.81.
    """

    def __init__(self, L1: float, L2: float, m1: float, m2: float, g: float = 9.81):
        if L1 <= 0 or L2 <= 0:
            raise ValueError("Длины маятников L1 и L2 должны быть положительными числами.")
        if m1 <= 0 or m2 <= 0:
            raise ValueError("Массы maятников m1 и m2 должны быть положительными числами.")

        self.L1 = float(L1)
        self.L2 = float(L2)
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.g = float(g)

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        """
        Вычисляет правые части дифференциальных уравнений двойного маятника.

        Параметры:
            state (np.ndarray): вектор состояния [theta1, theta2, omega1, omega2],
                                где theta — углы (рад), omega — угловые скорости (рад/с).

        Возвращает:
            dstate (np.ndarray): вектор производных [omega1, omega2, alpha1, alpha2].
        """
        theta1, theta2, omega1, omega2 = state
        Δ = theta1 - theta2

        sinΔ = np.sin(Δ)
        cosΔ = np.cos(Δ)
        denom1 = self.L1 * (self.m1 + self.m2 * sinΔ * sinΔ)
        denom2 = self.L2 * (self.m1 + self.m2 * sinΔ * sinΔ)

        # Предохраняемся от деления на ноль: если знаменатель слишком мал, добавляем eps
        eps = 1e-8
        if abs(denom1) < eps or abs(denom2) < eps:
            # Вырожденное состояние (почти вертикально расположены оба звена)
            # Возвращаем нулевые ускорения, чтобы интегратор не "взрывался"
            alpha1 = 0.0
            alpha2 = 0.0
        else:
            # Вычисление α₁ по формулам Лагранжа
            num1 = (self.m2 * self.g * np.sin(theta2) * cosΔ
                    - self.m2 * sinΔ * (self.L1 * omega1 ** 2 * cosΔ + self.L2 * omega2 ** 2)
                    - (self.m1 + self.m2) * self.g * np.sin(theta1))
            alpha1 = num1 / denom1

            # Вычисление α₂ по формулам Лагранжа
            num2 = ((self.m1 + self.m2) * (self.L1 * omega1 ** 2 * sinΔ
                                           - self.g * np.sin(theta2)
                                           + self.g * np.sin(theta1) * cosΔ)
                    + self.m2 * self.L2 * omega2 ** 2 * sinΔ * cosΔ)
            alpha2 = num2 / denom2

        return np.array([omega1, omega2, alpha1, alpha2], dtype=float)

    def integrate(self,
                  y0: np.ndarray,
                  t_max: float,
                  dt: float,
                  method: str = "rk4") -> (np.ndarray, np.ndarray):
        """
        Интегрирует систему уравнений двойного маятника на отрезке времени [0, t_max] с шагом dt.

        Параметры:
            y0      (np.ndarray): начальный вектор состояния [theta1_0, theta2_0, omega1_0, omega2_0].
            t_max   (float):       максимальное время моделирования (с).
            dt      (float):       шаг по времени (с).
            method  (str):         метод интегрирования. Пока поддерживается только "rk4".

        Возвращает:
            t (np.ndarray): массив времён от 0 до t_max с шагом dt, размер (N,).
            Y (np.ndarray): массив состояний размером (N, 4), где по столбцам:
                            [theta1, theta2, omega1, omega2].
        """
        if dt <= 0:
            raise ValueError("Шаг dt должен быть положительным числом.")
        if t_max <= 0:
            raise ValueError("t_max должно быть положительным числом.")
        if y0.shape != (4,):
            raise ValueError("Начальный вектор y0 должен быть размерности (4,) — [θ1, θ2, ω1, ω2].")

        t = np.arange(0.0, t_max + dt / 2, dt)  # Добавляем dt/2, чтобы включить t_max
        N = t.shape[0]
        Y = np.zeros((N, 4), dtype=float)
        Y[0, :] = y0

        if method.lower() == "rk4":
            for i in range(N - 1):
                dt_i = dt
                yi = Y[i, :]

                k1 = self.derivatives(yi)
                k2 = self.derivatives(yi + 0.5 * dt_i * k1)
                k3 = self.derivatives(yi + 0.5 * dt_i * k2)
                k4 = self.derivatives(yi + dt_i * k3)

                Y[i + 1, :] = yi + (dt_i / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise NotImplementedError(f"Метод интегрирования '{method}' не реализован. Только 'rk4'.")

        return t, Y

    def save_to_file(self, t: np.ndarray, Y: np.ndarray, filename: str = "pendulum_data.npz"):
        """
        Сохраняет результаты моделирования в файл .npz.

        Параметры:
            t (np.ndarray): одномерный массив времён.
            Y (np.ndarray): массив состояний (N, 4).
            filename (str): имя сохраняемого файла.
        """
        np.savez(filename, t=t, Y=Y)


if __name__ == "__main__":
    # Пример использования при запуске модуля напрямую.
    # Задаём параметры двойного маятника и интегрируем.
    pend = DoublePendulum(L1=2.0, L2=2.0, m1=1.0, m2=3.0)
    theta1_0 = np.pi / 2
    theta2_0 = np.pi / 2
    omega1_0 = 3.0
    omega2_0 = 0.0
    y0 = np.array([theta1_0, theta2_0, omega1_0, omega2_0], dtype=float)

    t_max = 25.0
    dt = 0.03
    t, Y = pend.integrate(y0=y0, t_max=t_max, dt=dt, method="rk4")
    print(f"Интегрирование завершено. Векторы t.shape = {t.shape}, Y.shape = {Y.shape}")
