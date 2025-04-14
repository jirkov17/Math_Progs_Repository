import numpy as np
from scipy.linalg import lu_factor, lu_solve


# Функция для формирования матрицы B
def create_matrix_B(n, p):
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            B[i, j] = 1 / (p + i + j + 1)  # i и j начинаются с 0, поэтому +1
    return B


# Функция для вычисления нормы матрицы
def matrix_norm(A):
    return np.linalg.norm(A, ord=2)


# Параметр p
p = 4

# Размерности матриц
dimensions = [4, 6, 8, 10, 12]

# Анализ для каждой размерности
for n in dimensions:
    # Формируем матрицу B
    B = create_matrix_B(n, p)

    # LU-декомпозиция (аналог DECOMP)
    lu, piv = lu_factor(B)

    # Вычисление обратной матрицы B^{-1} с использованием SOLVE
    E = np.eye(n)  # Единичная матрица
    B_inv = np.zeros_like(B)
    for i in range(n):
        B_inv[:, i] = lu_solve((lu, piv), E[:, i])  # Решение системы B * x = e_i

    # Вычисляем матрицу невязки R = BB^{-1} - E
    R = np.dot(B, B_inv) - E

    # Вычисляем число обусловленности
    cond_B = np.linalg.cond(B)

    # Вычисляем норму матрицы невязки
    norm_R = matrix_norm(R)

    # Вывод результатов
    print(f"n = {n}:")
    print(f"Число обусловленности cond(B) = {cond_B}")
    print(f"Норма матрицы невязки ||R|| = {norm_R}")
    print("-" * 40)
