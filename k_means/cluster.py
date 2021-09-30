import os
import uuid
from dataclasses import dataclass
from typing import List, Optional
import imageio
import matplotlib.pyplot as plt
import numpy as np

plt.interactive(False)


@dataclass
class Point:
    x: float
    y: float
    cluster: int = -1

    def dist(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def tuple(self):
        return self.x, self.y

    def __eq__(self, other) -> bool:
        return abs(self.x - other.x) <= 0.00001 and abs(self.y - other.y) <= 0.00001


@dataclass
class KMeans:
    points: List[Point]
    k: int
    centroids: List[Point] = None
    show_iteration: bool = False
    max_iterations: int = 100

    @property
    def inertia(self) -> Optional[float]:
        # Это функция J()
        _inertia = 0
        for centroid in self.centroids:
            cluster = list(filter(lambda p: p.cluster == centroid.cluster, self.points))
            _inertia += sum(map(lambda point: point.dist(centroid), cluster)) ** 2
        return _inertia

    def setup(self):
        x_center: float = np.mean(list(map(lambda p: p.x, self.points)))
        y_center: float = np.mean(list(map(lambda p: p.y, self.points)))
        center = Point(x_center, y_center)
        radius = max(map(lambda p: p.dist(center), self.points))
        x_c = [x_center + radius * np.cos(2 * np.pi * j / self.k) for j in range(self.k)]
        y_c = [y_center + radius * np.sin(2 * np.pi * j / self.k) for j in range(self.k)]
        self.centroids = [Point(x_c[i], y_c[i], i) for i in range(self.k)]
        self.nearest_centroid()

    def nearest_centroid(self) -> None:
        for point in self.points:
            temp = [(centroid, point.dist(centroid)) for centroid in self.centroids]
            nearest_centroid = min(temp, key=lambda ttuple: ttuple[1])
            point.cluster = nearest_centroid[0].cluster

    def recalculate_centroids(self):
        new_centroids = []
        for i in self.centroids:
            cluster = list(filter(lambda p: p.cluster == i.cluster, self.points))
            x_center: float = np.mean(list(map(lambda p: p.x, cluster)))
            y_center: float = np.mean(list(map(lambda p: p.y, cluster)))
            new_centroids.append(Point(x_center, y_center, i.cluster))
        return new_centroids

    def has_converged(self, new_centroid: List[Point]):
        self.centroids.sort(key=lambda x: x.tuple())
        new_centroid.sort(key=lambda x: x.tuple())
        for i in range(len(self.centroids)):
            if self.centroids[i] == new_centroid[i]:
                return True
        return False

    def fit(self):
        self.setup()
        show_clusters(title=f"Number of clusters: {self.k}, iteration: !{0}", points=self.points, centroids=self.centroids)
        new_centroids = self.recalculate_centroids()
        iteration = 1
        while (not self.has_converged(new_centroids)) and (iteration < self.max_iterations):
            self.centroids = new_centroids
            self.nearest_centroid()
            new_centroids = self.recalculate_centroids()
            if self.show_iteration:
                show_clusters(
                    title=f"Number of clusters: {self.k}, iteration: !{iteration}",
                    points=self.points,
                    centroids=self.centroids,
                )
            iteration += 1
        return self.points, self.centroids


def show_clusters(points: List[Point], centroids: List[Point], title=None):
    shade = 1.0 / (len(centroids) + 1)
    for i in centroids:
        cluster = list(filter(lambda p: p.cluster == i.cluster, points))
        plt.scatter(
            list(map(lambda p: p.x, cluster)),
            list(map(lambda p: p.y, cluster)),
            s=[2 for _ in range(len(cluster))],
            color=str(float(i.cluster + 1) * shade),
        )
    plt.scatter(list(map(lambda p: p.x, centroids)), list(map(lambda p: p.y, centroids)), color="r")
    plt.title(title)
    plt.savefig(f"plots/{len(centroids)}/{title or uuid.uuid4()}.png")
    plt.close()


def rand_points(n: int):
    return [Point(np.random.randint(0, 100), np.random.randint(0, 100)) for _ in range(n)]


def D(wcss: List[float]):
    result = []
    for i in range(1, len(wcss) - 1):
        result.append(abs(wcss[i] - wcss[i + 1]) / abs(wcss[i - 1] - wcss[i]))

    return result


def main():
    os.mkdir("plots")
    points = []
    # n = 100
    # points = rand_points(n)
    with open("datasets/s1.txt", "r") as file:
        for line in file.readlines():
            line = line.strip().split("    ")
            points.append(Point(int(line[0]), int(line[1])))

    wcss = [] # Это список всех J(k)
    k_means = []
    min_k = 1
    max_k = 20
    for k in range(min_k, max_k + 1):
        os.mkdir(f"plots/{k}", )
        k_mean = KMeans(points.copy(), k=k, show_iteration=True)
        k_means.append(k_mean)
        points, centroids = k_mean.fit()
        show_clusters(title=f"Number of clusters: {k}, iteration: !{301}", points=points, centroids=centroids)
        wcss.append(k_mean.inertia)
    plt.plot(wcss, linestyle='solid')
    plt.xlabel("K Numbers")
    plt.ylabel("Inertia")
    plt.savefig('plots/J.png')
    plt.close()
    result = D(wcss)
    plt.plot(list(range(1, max_k)), [0.0] + result)
    plt.xlabel("K Numbers")
    plt.ylabel("D")
    plt.savefig('plots/D.png')
    plt.close()
    list_D = [(i + 2, result[i]) for i in range(max_k - 2)]
    print(f"Optimal number of clustres: {min(list_D, key=lambda t: t[1])}")
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


if __name__ == "__main__":
    main()
