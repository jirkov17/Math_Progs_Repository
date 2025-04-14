import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

# Определение системы уравнений
def system(t, x):
    x1, x2 = x
    dx1dt = -130 * x1 + 900 * x2 + np.exp(-10 * t)
    dx2dt = 30 * x1 - 300 * x2 + np.log(1 + 100 * t**2)
    return [dx1dt, dx2dt]

# Начальные условия
x0 = [3, 0]  # x2(0) = -t, но при t=0, x2(0) = 0

# Шаг печати
h_print = 0.0075

# Временные точки для вывода результатов
t_eval = np.arange(0, 0.15 + h_print, h_print)

# Решение методом RKF45
sol_rkf45 = solve_ivp(system, [0, 0.15], x0, method='RK45', t_eval=t_eval, atol=1e-6, rtol=1e-6)

# Метод Рунге-Кутты 3-й степени
def runge_kutta_3rd_order(f, x0, t0, t_end, h):
    t = np.arange(t0, t_end + h, h)
    x = np.zeros((len(x0), len(t)))
    x[:, 0] = x0
    for i in range(1, len(t)):
        k1 = h * np.array(f(t[i-1], x[:, i-1]))
        k2 = h * np.array(f(t[i-1] + h/2, x[:, i-1] + k1/2))
        k3 = h * np.array(f(t[i-1] + 3*h/4, x[:, i-1] + 3*k2/4))
        x[:, i] = x[:, i-1] + (2*k1 + 3*k2 + 4*k3) / 9
    return t, x

# Шаги интегрирования
h_int_a = 0.0075
h_int_b = 0.00375  # Пример другого шага

# Решение методом Рунге-Кутты 3-й степени с шагом h_int_a
t_rk3_a, x_rk3_a = runge_kutta_3rd_order(system, x0, 0, 0.15, h_int_a)

# Решение методом Рунге-Кутты 3-й степени с шагом h_int_b
t_rk3_b, x_rk3_b = runge_kutta_3rd_order(system, x0, 0, 0.15, h_int_b)

# Интерполяция результатов для шага печати
x_rk3_a_interp = np.array([np.interp(t_eval, t_rk3_a, x_rk3_a[i, :]) for i in range(2)])
x_rk3_b_interp = np.array([np.interp(t_eval, t_rk3_b, x_rk3_b[i, :]) for i in range(2)])

# Создание таблиц с результатами
results_rkf45 = pd.DataFrame({
    't': t_eval,
    'x1_rkf45': sol_rkf45.y[0],
    'x2_rkf45': sol_rkf45.y[1]
})

results_rk3_a = pd.DataFrame({
    't': t_eval,
    'x1_rk3_a': x_rk3_a_interp[0],
    'x2_rk3_a': x_rk3_a_interp[1]
})

results_rk3_b = pd.DataFrame({
    't': t_eval,
    'x1_rk3_b': x_rk3_b_interp[0],
    'x2_rk3_b': x_rk3_b_interp[1]
})

# Вывод таблиц
print("Метод RKF45:")
print(results_rkf45)

print("\nМетод Рунге-Кутты 3-й степени (h=0.0075):")
print(results_rk3_a)

print("\nМетод Рунге-Кутты 3-й степени (h=0.00375):")
print(results_rk3_b)

# Построение графиков для RKF45
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t_eval, sol_rkf45.y[0], label='x1 RKF45')
plt.plot(t_eval, sol_rkf45.y[1], label='x2 RKF45')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.title('Решение системы методом RKF45')

plt.tight_layout()
plt.show()

# Построение графиков для метода Рунге-Кутты 3-й степени при h=0.0075
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t_eval, x_rk3_a_interp[0], label='x1 RK3 h=0.0075')
plt.plot(t_eval, x_rk3_a_interp[1], label='x2 RK3 h=0.0075')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.title('Решение системы методом Рунге-Кутты 3-й степени (h=0.0075)')
plt.tight_layout()
plt.show()

# Построение графиков для метода Рунге-Кутты 3-й степени при h=0.00375
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t_eval, x_rk3_b_interp[0], label='x1 RK3 h=0.00375')
plt.plot(t_eval, x_rk3_b_interp[1], label='x2 RK3 h=0.00375')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.title('Решение системы методом Рунге-Кутты 3-й степени (h=0.00375)')

plt.tight_layout()
plt.show()
