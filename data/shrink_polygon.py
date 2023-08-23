import numpy as np
import matplotlib.pyplot as plt

# 多边形周长
# shape of polygon: [N, 2]
def Perimeter(polygon: np.array):
    N, d = polygon.shape
    if N < 3 or d != 2:
        raise ValueError

    permeter = 0.
    for i in range(N):
        permeter += np.linalg.norm(polygon[i-1] - polygon[i])
    return permeter


# 面积
def Area(polygon: np.array):
    N, d = polygon.shape
    if N < 3 or d != 2:
        raise ValueError

    area = 0.
    vector_1 = polygon[1] - polygon[0]
    for i in range(2, N):
        vector_2 = polygon[i] - polygon[0]
        area += np.abs(np.cross(vector_1, vector_2))
        vector_1 = vector_2
    return area / 2

# |r| < 1
# r > 0, 内缩
# r < 0, 外扩
def calc_shrink_width(polygon: np.array, r):
    area = Area(polygon)
    perimeter = Perimeter(polygon)
    L = area * (1 - r ** 2) / perimeter
    return L if r > 0 else -L


def shrink_polygon(polygon: np.array, r):
    N, d = polygon.shape
    if N < 3 or d != 2:
        raise ValueError

    shrinked_polygon = []
    L = calc_shrink_width(polygon, r)
    for i in range(N):
        Pi = polygon[i]
        v1 = polygon[i-1] - Pi
        v2 = polygon[(i+1)%N] - Pi

        normalize_v1 = v1 / np.linalg.norm(v1)
        normalize_v2 = v2 / np.linalg.norm(v2)

        sin_theta = np.abs(np.cross(normalize_v1, normalize_v2))

        Qi = Pi + L / sin_theta * (normalize_v1 + normalize_v2)
        shrinked_polygon.append(Qi)
    return np.asarray(shrinked_polygon)




if __name__ == "__main__":
    poly = np.array([[0, 0], [0, 1], [0.5, 2], [1, 1], [1, 0]])
    perimeter = Perimeter(poly)
    area = Area(poly)

    shrink_poly = shrink_polygon(poly, 0.95)
    expansion_poly = shrink_polygon(shrink_poly, -0.5)
    plt.scatter(poly[:, 0], poly[:, 1], c='r')
    plt.scatter(shrink_poly[:, 0], shrink_poly[:, 1], c='y')

    plt.show()
    print(perimeter, area, shrink_poly, expansion_poly)
