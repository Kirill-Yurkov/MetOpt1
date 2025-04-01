import numpy as np
import matplotlib.pyplot as plt

start = [0, 0]


def function_himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def gradient(x, y):
    df_dx = 4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7)
    df_dy = 2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7)
    return np.array([df_dx, df_dy])


def hessian(x, y):
    d2f_dx2 = 12 * x ** 2 + 4 * y - 42
    d2f_dy2 = 4 * x + 12 * y ** 2 - 26
    d2f_dxdy = 4 * (x + y)
    return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])


def gradient_descent_variable_step(start):
    point = np.array(start, dtype=float)
    path = [point.copy()]
    for i in range(10000):
        grad = gradient(point[0], point[1])
        hess = hessian(point[0], point[1])
        hess_norm = np.linalg.norm(hess, 2) 
        if hess_norm == 0:
            break  
        h = 1 / hess_norm  
        if np.linalg.norm(grad) < 0.001:
            break
        point -= h * grad
        path.append(point.copy())
    return point, np.array(path)


min_point, path = gradient_descent_variable_step(start)
print(f"Найденный минимум: {np.round(min_point, 4)}")

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = function_himmelblau(X, Y)
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(X, Y)')
ax.set_title('Функция Химмельблау')
plt.show()


plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Контурный график тестовой функции')
plt.plot(path[:, 0], path[:, 1], color='red', marker='o', markersize=1, label='Путь градиентного спуска')
plt.scatter(min_point[0], min_point[1], color='orange', marker='x', s=50, label='Найденный минимум')
plt.scatter(start[0], start[1], color='orange', marker='o', s=50, label='Начальная точка')
plt.legend()
plt.show()
