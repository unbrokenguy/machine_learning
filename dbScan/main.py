"""
ЛКМ - поставить точку и ее соседей
r - (английская) раскарсить зеленые, желтые и красные точки.
t - (английская) найти и раскарсить кластеры.
"""
from dataclasses import dataclass, field
from enum import Enum
import random
from typing import List

import numpy as np
import pygame


R = 4
K = 4
RUNNING = True
MIN_NEIGHBORS = 3
EPSILON = 15
FPS = 5


class Color(Enum):
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    YELLOW = (255, 255, 0)


def generate_colors(n: int):
    colors = []
    for _ in range(n):
        hex_color = "%06x" % random.randint(0, 0xFFFFFF)
        colors.append(tuple(int(hex_color[i: i + 2], 16) for i in (0, 2, 4)))
    return colors


@dataclass
class Point:
    x: int
    y: int
    cluster: int = 0
    color: Color = field(init=False)

    def __post_init__(self):
        self.color = Color.RED

    def dist(self, other):
        if not isinstance(other, Point):
            return ValueError
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Point):
            return False
        if self is other:
            return True
        return abs(self.x - other.x) <= 0.00001 and abs(self.y - other.y) <= 0.00001


class DbScan:
    def __init__(self, points: Point = None):
        self.points = points or []

    def add_point(self, point: Point):
        self.points.append(point)
        self._generate_neighbors(point)

    def find_all_by_color(self, color: Color):
        return list(filter(lambda p: p.color == color, self.points))

    def set_cluster(self, point: Point, points: List[Point]):
        for neighbor in points:
            if neighbor.cluster == 0 and point.dist(neighbor) <= EPSILON:
                neighbor.cluster = point.cluster
                self.set_cluster(neighbor, points)

    def _generate_neighbor(self, point: Point) -> None:
        d = random.randint(2 * R, 5 * R)
        alpha = random.random() * np.pi
        self.points.append(
            Point(point.x + d * np.sin(alpha), point.y + d * np.cos(alpha))
        )

    def _generate_neighbors(self, point: Point):
        k = random.randint(1, K)
        for _ in range(k):
            self._generate_neighbor(point)

    def _clear(self):
        for p in self.points:
            p.color = Color.RED
            p.cluster = 0

    def set_colors(self):
        self._clear()
        self._color_green_points()
        self._color_yellow_points()

    def _color_green_points(self) -> None:
        for point in self.points:
            neighbour_count = 0
            for neighbour in self.points:
                if point == neighbour:
                    continue
                neighbour_count += 1 if point.dist(neighbour) <= EPSILON else 0
            if neighbour_count >= MIN_NEIGHBORS:
                point.color = Color.GREEN

    def _color_yellow_points(self):
        red_points = self.find_all_by_color(Color.RED)
        for point in red_points:
            for neighbour in self.points:
                if neighbour.color == Color.GREEN:
                    point.color = (
                        Color.YELLOW
                        if point.dist(neighbour) <= EPSILON
                        else point.color
                    )

    def make_clusters(self):
        green_points = self.find_all_by_color(Color.GREEN)
        cluster_count = 1
        for green in green_points:
            if green.cluster == 0:
                green.cluster = cluster_count
                cluster_count += 1
                self.set_cluster(green, green_points)

        yellow_points = self.find_all_by_color(Color.YELLOW)

        for yellow in yellow_points:
            min_dist = yellow.dist(green_points[0])
            yellow.cluster = green_points[0].cluster
            for i in range(1, len(green_points)):
                if yellow.dist(green_points[i]) < min_dist:
                    min_dist = yellow.dist(green_points[i])
                    yellow.cluster = green_points[i].cluster
        cluster_colors = generate_colors(cluster_count)
        print(cluster_count)
        for point in list(filter(lambda p: p.cluster != 0, self.points)):
            point.color = cluster_colors[point.cluster - 1]


def main():
    db_scan = DbScan()
    pygame.init()
    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    screen.fill("WHITE")
    pygame.display.update()
    clock = pygame.time.Clock()
    play = True
    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    db_scan.add_point(Point(x=event.pos[0], y=event.pos[1]))
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    db_scan.set_colors()
                if event.key == pygame.K_t:
                    db_scan.set_colors()
                    db_scan.make_clusters()
        screen.fill("WHITE")
        for point in db_scan.points:
            pygame.draw.circle(
                screen,
                point.color.value if isinstance(point.color, Color) else point.color,
                (point.x, point.y),
                R,
            )
        pygame.display.update()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
