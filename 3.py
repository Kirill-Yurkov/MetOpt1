import numpy as np
import matplotlib.pyplot as plt

start = [0, 0]
h = 0.0005  
momentum = 0.9

def function_himmelblau(x, y):
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def gradient(x, y):
    df_dx = 4 * x * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7)
    df_dy = 2 * (x ** 2 + y - 11) + 4 * y * (x + y ** 2 - 7)
    return np.array([df_dx, df_dy])


def nesterov_gradient_descent(start, h, momentum):
    point = np.array(start, dtype=float)
    velocity = np.zeros_like(point)
    path = [point.copy()]
    for i in range(10000):
        lookahead_point = point + momentum * velocity
        grad = gradient(lookahead_point[0], lookahead_point[1])
        if np.linalg.norm(grad) < 0.001:
            break
        velocity = momentum * velocity - h * grad
        point += velocity
        path.append(point.copy())
    return point, np.array(path)

min_point, path = nesterov_gradient_descent(start, h, momentum)
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

#моментум - накопленная скорость, он определяет, насколько сильно учитывается предыдущее направление движения, 
# выбрано 0.9, во-первых, потому что самое часто встречающееся в литературе, во-вторых, при слишком малых значениях 
# (<0.5), метод не даст нужного ускорения, и наоборот при слишком больших (>0.95) есть вероятность того, 
# что метод пропустит нужную точку минимума и есть вероятность колебаний, что может снизить точность 
#метод отличается от обычного метода градиентного спуска тем, что мы считаем градиент не в текущей точке, 
# а в предсказанной lookahead_point (строчка 22)