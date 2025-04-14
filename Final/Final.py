
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Параметры задачи
epsilon = 0.999999051
l = 0.999998351

A = 0
B = 1


def solve_heat_transfer(n):
    h = l / n  # шаг сетки

    # Создаем матрицу системы и правую часть
    size = n - 1  # количество внутренних точек
    matrix = np.zeros((size, size))
    rhs = np.zeros(size)

    # Заполняем матрицу и правую часть
    for i in range(size):
        x_i = (i + 1) * h  # x_i = x_0 + i*h, x_0=0

        # Коэффициенты для T_{i-1}, T_i, T_{i+1}
        a = epsilon - h * x_i / 2
        b = -2 * epsilon - h ** 2 * x_i
        c = epsilon + h * x_i / 2

        # Заполняем матрицу
        if i > 0:
            matrix[i, i - 1] = a
        matrix[i, i] = b
        if i < size - 1:
            matrix[i, i + 1] = c

    # Учет граничных условий
    # Первое уравнение (i=1) имеет T_0 = A
    matrix[0, 0] = b
    matrix[0, 1] = c
    rhs[0] = -a * A  # переносим известное значение в правую часть

    # Последнее уравнение (i=n-1) имеет T_n = B
    matrix[-1, -2] = a
    matrix[-1, -1] = b
    rhs[-1] = -c * B  # переносим известное значение в правую часть

    # Решаем систему
    T_inner = solve(matrix, rhs)

    # Добавляем граничные значения
    T = np.zeros(n + 1)
    T[0] = A
    T[-1] = B
    T[1:-1] = T_inner

    # Создаем массив x
    x = np.linspace(0, l, n + 1)

    return x, T


# Решаем для n=10 и n=20
x10, T10 = solve_heat_transfer(10)
x20, T20 = solve_heat_transfer(20)

# Выводим таблицы значений
print("Решение для n=10:")
print(f"{'x':<15}{'T(x)':<15}")
for xi, Ti in zip(x10, T10):
    print(f"{xi:.10f}\t{Ti:.10f}")

print("\nРешение для n=20:")
print(f"{'x':<15}{'T(x)':<15}")
for xi, Ti in zip(x20, T20):
    print(f"{xi:.10f}\t{Ti:.10f}")

# Вычисляем разность в общих точках
common_x = x10  # Точки для n=10
# Находим соответствующие значения для n=20 через интерполяцию
T20_at_x10 = np.interp(x10, x20, T20)
difference = T10 - T20_at_x10

# Строим графики
plt.figure(figsize=(12, 8))

# График решений
plt.subplot(2, 1, 1)
plt.plot(x10, T10, 'o-', label='n=10')
plt.plot(x20, T20, 's-', label='n=20', markersize=3)
plt.xlabel('x')
plt.ylabel('T(x)')
plt.title('Решение краевой задачи теплопередачи')
plt.legend()
plt.grid(True)

# График разности
plt.subplot(2, 1, 2)
plt.plot(x10, difference, 'r*-', label='Разность (n=10 - n=20)')
plt.xlabel('x')
plt.ylabel('ΔT(x)')
plt.title('Разность между решениями для n=10 и n=20')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Оценка погрешности
max_error = np.max(np.abs(difference))
print(f"\nМаксимальная разница между решениями: {max_error:.10f}")
print(f"Средняя абсолютная разница: {np.mean(np.abs(difference)):.10f}")

