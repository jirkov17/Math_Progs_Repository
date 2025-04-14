
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Параметры задачи
epsilon = 0.999999051
l = 0.999998351

A = 0
B = 1

def solve_heat_transfer(n):
    h = l / n
    
    size = n - 1
    matrix = np.zeros((size, size))
    rhs = np.zeros(size)
    
    for i in range(size):
        x_i = (i + 1) * h
        
        a = epsilon - h * x_i / 2
        b = -2 * epsilon - h**2 * x_i
        c = epsilon + h * x_i / 2
        
        if i > 0:
            matrix[i, i-1] = a
        matrix[i, i] = b
        if i < size - 1:
            matrix[i, i+1] = c
    
    matrix[0, 0] = b
    matrix[0, 1] = c
    rhs[0] = -a * A
    
    matrix[-1, -2] = a
    matrix[-1, -1] = b
    rhs[-1] = -c * B
    
    T_inner = solve(matrix, rhs)
    
    T = np.zeros(n + 1)
    T[0] = A
    T[-1] = B
    T[1:-1] = T_inner
    
    x = np.linspace(0, l, n + 1)
    
    return x, T

# Решаем для n=10 и n=20
x10, T10 = solve_heat_transfer(10)
x20, T20 = solve_heat_transfer(20)

# Вычисляем градиенты
def calculate_gradient(x, T):
    h = x[1] - x[0]
    dTdx = np.zeros_like(T)
    # Центральные разности для внутренних точек
    dTdx[1:-1] = (T[2:] - T[:-2]) / (2 * h)
    # Односторонние разности для границ
    dTdx[0] = (T[1] - T[0]) / h
    dTdx[-1] = (T[-1] - T[-2]) / h
    return dTdx

dTdx10 = calculate_gradient(x10, T10)
dTdx20 = calculate_gradient(x20, T20)

# Максимальные градиенты
max_grad10 = np.max(np.abs(dTdx10))
max_grad20 = np.max(np.abs(dTdx20))
print(f"Максимальный градиент (n=10): {max_grad10:.5f}")
print(f"Максимальный градиент (n=20): {max_grad20:.5f}")

# Строим графики
plt.figure(figsize=(12, 8))

# График температур
plt.subplot(2, 1, 1)
plt.plot(x10, T10, 'o-', label='n=10')
plt.plot(x20, T20, 's-', label='n=20', markersize=3)
plt.xlabel('x')
plt.ylabel('T(x)')
plt.title('Распределение температуры')
plt.legend()
plt.grid()

# График градиентов
plt.subplot(2, 1, 2)
plt.plot(x10, dTdx10, 'o-', label='n=10')
plt.plot(x20, dTdx20, 's-', label='n=20', markersize=3)
plt.xlabel('x')
plt.ylabel('dT/dx')
plt.title('Градиент температуры')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Разность решений
T20_interp = np.interp(x10, x20, T20)
difference = T10 - T20_interp

plt.figure(figsize=(10, 5))
plt.plot(x10, difference, 'r*-')
plt.xlabel('x')
plt.ylabel('ΔT(x)')
plt.title('Разность между решениями n=10 и n=20')
plt.grid()
plt.show()
