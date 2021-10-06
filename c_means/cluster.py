import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List
import imageio
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Point:
    x: float
    y: float

    def dist(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


@dataclass
class CMeans:
    points: List[Point]
    k: int
    m: float = 2
    epsilon: float = 0.001
    centroids: List[Point] = None
    show_iteration: bool = False
    u: List[List[float]] = None
    distances: List[List[float]] = None

    def __post_init__(self):
        # Инициализируем центроиды
        x_center = np.mean(list(map(lambda p: p.x, self.points)))
        y_center = np.mean(list(map(lambda p: p.y, self.points)))
        center = Point(x_center, y_center)
        radius = max(map(lambda p: p.dist(center), self.points))
        x_c = [x_center + radius * np.cos(2 * np.pi * j / self.k) for j in range(self.k)]
        y_c = [y_center + radius * np.sin(2 * np.pi * j / self.k) for j in range(self.k)]
        self.centroids = [Point(x_c[i], y_c[i]) for i in range(self.k)]
        # Вычисляем расстояние до центров кластеров
        self.distances = []
        for i in range(self.k):
            self.distances.append([])
            for j in range(len(self.points)):
                self.distances.append(self.points[j].dist(self.centroids[i]))

        # Задаем случайную матрицу принадлежности к кластерам
        random_matrix = np.random.random((self.k, len(self.points)))  # случайные значение от 0 до 1
        _sum = sum(random_matrix)  # суммирует вертикально
        # приводим к нормальному виду, чтобы сумма вероятностей принадлежности точек к кластерам была равна 1.0
        for i in range(self.k):
            for j in range(len(self.points)):
                random_matrix[i][j] = random_matrix[i][j] / _sum[j]
        self.u = random_matrix

    def calculate_centroids(self):
        # Пересчитываем центроиды
        # https://wikimedia.org/api/rest_v1/media/math/render/svg/c2f8602b082a7e8bd7d51b83bdb328898be77763
        for k in range(self.k):
            top_sum = [0.0, 0.0]
            bottom_sum = 0.0
            for i in range(len(self.points)):
                top_sum[0] += (self.u[k][i] ** self.m) * self.points[i].x
                top_sum[1] += (self.u[k][i] ** self.m) * self.points[i].y
                bottom_sum += self.u[k][i] ** self.m
            self.centroids[k] = Point(top_sum[0] / bottom_sum, top_sum[1] / bottom_sum)

    def recalculate_coefficients(self):
        # Пересчитываем матрицу коэффициентов принадлежности
        # Я запутался в формуле в документе, поэтому взял не упрощенную из английской википедии
        # https://wikimedia.org/api/rest_v1/media/math/render/svg/0072b0e3d088f0189660ff5cc29335399b28b0b7
        for i in range(self.k):
            for j in range(len(self.points)):
                denominator = 0.0
                for k in range(self.k):
                    denominator += (
                        self.points[j].dist(self.centroids[i]) / self.points[j].dist(self.centroids[k])
                    ) ** (2 / (self.m - 1))
                self.u[i][j] = 1.0 / denominator

    def has_converged(self, old_u):
        # Не важно сохранять структуру матрицы, чтобы узнать max(|Uij(r) - Uij(r-1)|)
        _temp_u = []
        for i in range(len(self.u)):
            for j in range(len(self.u[i])):
                _temp_u.append(abs(self.u[i][j] - old_u[i][j]))
        return max(_temp_u) < self.epsilon

    def fit(self):
        show_clusters(
            title=f"Number of clusters: {self.k}, iteration: !{0}", points=self.points, centroids=self.centroids
        )
        old_u = self.u.copy()
        self.recalculate_coefficients()
        iteration = 1
        while not self.has_converged(old_u):
            self.calculate_centroids()
            old_u = self.u.copy()
            self.recalculate_coefficients()
            if self.show_iteration:
                show_clusters(
                    title=f"Number of clusters: {self.k}, iteration: !{iteration}",
                    points=self.points,
                    centroids=self.centroids,
                )
            iteration += 1


def show_clusters(points: List[Point], centroids: List[Point], title=None):
    plt.scatter(
        list(map(lambda p: p.x, points)),
        list(map(lambda p: p.y, points)),
        s=[2 for _ in range(len(points))],
        color="0.2",
    )
    plt.scatter(list(map(lambda p: p.x, centroids)), list(map(lambda p: p.y, centroids)), color="r")
    plt.title(title)
    Path.mkdir(Path(f"plots/{len(centroids)}"), parents=True, exist_ok=True)
    plt.savefig(f"plots/{len(centroids)}/{title or uuid.uuid4()}.png")
    plt.close()


def rand_points(n: int):
    return [Point(np.random.randint(0, 1000), np.random.randint(0, 1000)) for _ in range(n)]


def make_gifs():
    for directory in os.listdir("plots"):
        if directory != "D.png" and directory != "J.png":
            images = []
            file_names = []
            for file in os.listdir(f"plots/{directory}"):
                file_names.append(file)
            file_names.sort(key=lambda s: int(s.split("!")[1].split(".")[0]))
            for f in file_names:
                for i in range(10):
                    images.append(imageio.imread(f"plots/{directory}/{f}"))
            imageio.mimsave(f"plots/{directory}.gif", images)


def main():
    points = []
    with open("datasets/s1.txt", "r") as file:
        for line in file.readlines():
            line = line.strip().split("    ")
            points.append(Point(int(line[0]), int(line[1])))

    c_means = CMeans(k=15, points=points, show_iteration=True)
    c_means.fit()
    make_gifs()


if __name__ == "__main__":
    main()
