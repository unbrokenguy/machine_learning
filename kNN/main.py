from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
import pygame
import random
from scipy.stats import mode


N = 3
R = 4
FPS = 5
POINTS_NUMBER = 10
MIN_NEIGHBOURS = 5
MAX_NEIGHBOURS = 15
OPTIMAL_NEIGHBOURS_COUNT = [0 for _ in range(MAX_NEIGHBOURS + 1)]


class Color(Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)
    BLACK = (0, 0, 0)


def generate_colors(n: int):
    colors = []
    for _ in range(n):
        hex_color = "%06x" % random.randint(0, 0xFFFFFF)
        colors.append(tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4)))
    return colors


def generate_points(clusters, points_number, colors):
    points = []
    for cluster in range(clusters):
        center_x, center_y = random.randint(50, 550), random.randint(50, 350)
        for element in range(points_number):
            points.append(Point(x=int(random.gauss(center_x, 20)), y=int(random.gauss(center_y, 20)),
                                cluster=cluster, color=colors[cluster]))
    return points


@dataclass
class Point:
    x: int
    y: int
    cluster: int
    color: Color = Color.BLACK

    def dist(self, other):
        if not isinstance(other, Point):
            return ValueError
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class KNN:

    def __init__(self, points, colors):
        self.points = points
        self.cluster_colors = colors

    def find_neighbours(self, point, k):
        return sorted(self.points, key=lambda p: point.dist(p))[:k]

    def add_point(self, point, cluster):
        point.cluster = cluster
        point.color = self.cluster_colors[cluster]
        for k in range(MIN_NEIGHBOURS, MAX_NEIGHBOURS):
            neighbors = self.find_neighbours(point, k)
            clusters = list(map(lambda p: p.cluster, neighbors))
            OPTIMAL_NEIGHBOURS_COUNT[k] = OPTIMAL_NEIGHBOURS_COUNT[k] + 1 if self.predict(point) == clusters else OPTIMAL_NEIGHBOURS_COUNT[k]
        self.points.append(point)

    def predict(self, point):
        optimal_cluster_number = 1 if max(OPTIMAL_NEIGHBOURS_COUNT) == 0 else OPTIMAL_NEIGHBOURS_COUNT.index(max(OPTIMAL_NEIGHBOURS_COUNT))
        neighbours = self.find_neighbours(point, optimal_cluster_number)
        count = Counter(list(map(lambda p: p.color, neighbours)))
        max_color = max(count.values())
        return list(count.keys())[list(count.values()).index(max_color)]


def main():
    colors = [Color.RED, Color.GREEN, Color.BLUE]
    points = generate_points(N, POINTS_NUMBER, colors)
    knn = KNN(points=points, colors=colors)
    pygame.init()
    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    screen.fill("WHITE")
    pygame.display.update()
    clock = pygame.time.Clock()
    play = True
    point = None
    while play:
        screen.fill("WHITE")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    x, y = pygame.mouse.get_pos()
                    point = Point(x, y, 0, Color.BLACK)
                if event.button == 3:
                    x, y = pygame.mouse.get_pos()
                    knn.points.append(Point(x, y, 0, knn.predict(Point(x, y, 0))))
                    point = None
            if event.type == pygame.KEYDOWN:
                cluster = 0
                if event.key == pygame.K_1:
                    point.color = colors[0]
                    cluster = 1
                if event.key == pygame.K_2:
                    point.color = colors[1]
                    cluster = 2
                if event.key == pygame.K_3:
                    point.color = colors[2]
                    cluster = 3
                if point is not None:
                    knn.add_point(point, cluster)
                    point = None
        if point:
            pygame.draw.circle(
                screen,
                point.color.value if isinstance(point.color, Color) else point.color,
                (point.x, point.y),
                R,
            )
        for p in knn.points:
            pygame.draw.circle(
                screen,
                p.color.value if isinstance(p.color, Color) else p.color,
                (p.x, p.y),
                R,
            )
        pygame.display.update()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
