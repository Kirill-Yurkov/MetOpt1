import numpy as np
import matplotlib.pyplot as plt

def test_function(x, y):
    """
    Тестовая функция, заданная преподавателем
    """
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2  # Замените на нужную функцию

def gradient(x, y):
    """
    Градиент тестовой функции
    """
    df_dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
    df_dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
    return np.array([df_dx, df_dy])

def gradient_descent(start, lr=0.0005, tol=1e-6, max_iter=10000):
    """
    Градиентный спуск для поиска минимума с постоянным шагом
    """
    point = np.array(start, dtype=float)
    for _ in range(max_iter):
        grad = gradient(point[0], point[1])
        if np.linalg.norm(grad) < tol:
            break
        point -= lr * grad
    return point

# Поиск минимума
min_point = gradient_descent(start=[0, 0])
print(f"Найденный минимум: {min_point}")

# Создание сетки значений
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = test_function(X, Y)

# Построение контурного графика
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Контурный график тестовой функции')
plt.scatter(min_point[0], min_point[1], color='red', marker='x', s=100, label='Найденный минимум')
plt.legend()
plt.show()
