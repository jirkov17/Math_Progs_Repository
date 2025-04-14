
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from mpmath import mp

# Устанавливаем точность (количество знаков после запятой)
mp.dps = 15  # Запас для точности 10 знаков

# Вычисление ε (эпсилон) с повышенной точностью
def integrand(x):
    return np.sqrt(3 - x) * np.cos(x)

epsilon, _ = quad(integrand, 0, np.pi/2)
epsilon = epsilon * 0.6436369
print(f"ε = {epsilon:.10f}")

# Вычисление ℓ (эль) с повышенной точностью
def equation(x):
    return 1 / np.tan(x) + x / (1 - x**2)

# Поиск наименьшего положительного корня (x*)
result = root_scalar(equation, bracket=(4.0, 6.0), method='brentq')
x_star = result.root
ell = 0.2231271 * x_star
print(f"ℓ = {ell:.10f}")

