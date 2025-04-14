import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Определение системы дифференциальных уравнений
def system(t, x):
    x1, x2 = x
    dx1dt = -130 * x1 + 900 * x2 + np.exp(-10 * t)
    dx2dt = 30 * x1 - 300 * x2 + np.log(1 + 100 * t**2)
    return [dx1dt, dx2dt]

# Начальные условия
x0 = [3, 0]  # x2(0) = -t, но t=0, поэтому x2(0) = 0

# Временной интервал
t_span = (0, 0.15)

# Метод RKF45
sol_rkf45 = solve_ivp(system, t_span, x0, method='RK45', t_eval=np.arange(0, 0.15, 0.0075), atol=1e-4, rtol=1e-4)

# Реализация метода Рунге-Кутты 3-й степени
def runge_kutta_3rd_order(system, t_span, x0, h):
    t0, tf = t_span
    t = np.arange(t0, tf, h)
    x = np.zeros((len(t), len(x0)))
    x[0] = x0

    for i in range(1, len(t)):
        tn = t[i-1]
        xn = x[i-1]
        k1 = np.array(system(tn, xn)) * h
        k2 = np.array(system(tn + h/2, xn + k1/2)) * h
        k3 = np.array(system(tn + 3*h/4, xn + 3*k2/4)) * h
        x[i] = xn + (2*k1 + 3*k2 + 4*k3) / 9

    return t, x

# Шаг интегрирования
h = 0.0075

# Решение методом Рунге-Кутты 3-й степени
t_rk3, x_rk3 = runge_kutta_3rd_order(system, t_span, x0, h)

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.plot(sol_rkf45.t, sol_rkf45.y[0], 'b-', label='RKF45 x1')
plt.plot(sol_rkf45.t, sol_rkf45.y[1], 'g-', label='RKF45 x2')
plt.plot(t_rk3, x_rk3[:, 0], 'r--', label='RK3 x1')
plt.plot(t_rk3, x_rk3[:, 1], 'm--', label='RK3 x2')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.title('Comparison of RKF45 and 3rd Order Runge-Kutta Methods')
plt.grid(True)
plt.show()
