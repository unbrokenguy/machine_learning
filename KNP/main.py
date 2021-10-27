import random
from copy import deepcopy
from dataclasses import dataclass
from typing import List
from PIL import Image, ImageDraw, ImageFont
import numpy as np

WIDTH = 1100
SIZE = (0, WIDTH)
N = 6
K = 3


@dataclass
class Peak:
    x: int
    y: int

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))


@dataclass
class Edge:
    first: Peak
    second: Peak
    weight: int = 0

    def __eq__(self, other):
        return (
            self.first == other.first
            and self.second == other.second
            or self.first == other.second
            and self.second == other.first
        )


@dataclass
class Graph:
    peaks: List[Peak]
    edges: List[Edge]


def generate_random_graph(peaks_count: int):
    radius = WIDTH / 2
    x_c = [
        SIZE[1] // 2 + radius * np.cos(2 * np.pi * j / peaks_count)
        for j in range(peaks_count)
    ]
    y_c = [
        SIZE[1] // 2 + radius * np.sin(2 * np.pi * j / peaks_count)
        for j in range(peaks_count)
    ]
    peaks = [Peak(int(x_c[i]), int(y_c[i])) for i in range(peaks_count)]
    edges = []
    for i in range(len(peaks)):
        for j in range(i, len(peaks)):
            if peaks[i] == peaks[j]:
                continue
            edges.append(Edge(peaks[i], peaks[j], random.randint(*SIZE)))
    return Graph(peaks, edges)


def find_min_edge_to_min_path(min_path, left_edges):
    left_edges.sort(key=lambda e: e.weight)
    if len(min_path) == 0:
        return left_edges[0]
    insulated_peaks = []
    for e in min_path:
        insulated_peaks.append(e.first)
        insulated_peaks.append(e.second)

    for i in range(len(left_edges)):
        if (
            left_edges[i].first in insulated_peaks
            or left_edges[i].second in insulated_peaks
        ):
            return left_edges[i]


def find_max_edge(edges: List[Edge]):
    return max(edges, key=lambda x: x.weight)


def knp(graph: Graph):
    min_path = []
    left_edges = deepcopy(graph.edges)
    while True:
        insulated_peaks = set()
        for e in min_path:
            insulated_peaks.add(e.first)
            insulated_peaks.add(e.second)
        if len(insulated_peaks) == len(graph.peaks):
            break
        edge = find_min_edge_to_min_path(min_path, left_edges)
        left_edges.remove(edge)
        min_path.append(edge)
    make_clusters(min_path)


def make_clusters(edges):
    edges.sort(key=lambda e: -1 * e.weight)
    im, d = get_peaks_image(edges)
    draw(edges[K - 1 :], im=im, d=d)


def get_peaks_image(edges: List[Edge]):
    im = Image.new("RGBA", (WIDTH, WIDTH), (255, 255, 255, 255))
    d = ImageDraw.Draw(im)
    for edge in edges:
        d.ellipse(
            (
                edge.first.x - 15,
                edge.first.y - 15,
                edge.first.x + 15,
                edge.first.y + 15,
            ),
            fill=(255, 0, 0),
            outline=(0, 0, 0),
        )
        d.ellipse(
            (
                edge.second.x - 15,
                edge.second.y - 15,
                edge.second.x + 15,
                edge.second.y + 15,
            ),
            fill=(255, 0, 0),
            outline=(0, 0, 0),
        )
    return im, d


def draw(edges: List[Edge], im=None, d=None):
    im = im or Image.new("RGBA", (WIDTH, WIDTH), (255, 255, 255, 255))
    d = d or ImageDraw.Draw(im)
    font = ImageFont.truetype("arial.ttf", size=20)
    for edge in edges:
        d.line(
            (edge.first.x, edge.first.y, edge.second.x, edge.second.y),
            fill=128,
            width=5,
        )
        d.text(
            (
                (edge.first.x + edge.second.x) // 2 + random.randint(-30, 30),
                (edge.first.y + edge.second.y) // 2 + random.randint(-30, 30),
            ),
            str(edge.weight),
            fill=(255, 0, 0),
            font=font,
        )
    for edge in edges:
        d.ellipse(
            (
                edge.first.x - 15,
                edge.first.y - 15,
                edge.first.x + 15,
                edge.first.y + 15,
            ),
            fill=(255, 0, 0),
            outline=(0, 0, 0),
        )
        d.ellipse(
            (
                edge.second.x - 15,
                edge.second.y - 15,
                edge.second.x + 15,
                edge.second.y + 15,
            ),
            fill=(255, 0, 0),
            outline=(0, 0, 0),
        )
    im.show()


if __name__ == "__main__":
    graph = generate_random_graph(N)
    print(graph.edges)
    draw(graph.edges)
    knp(graph)
