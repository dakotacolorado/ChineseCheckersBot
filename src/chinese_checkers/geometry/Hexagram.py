from typing import List, Tuple, Dict

from .Vector import Vector


class Hexagram:

    radius: int
    hexagon_points: List[Vector]
    hexagon_edges: List[Tuple[Vector, Vector]]
    rhombus_grid: List[Vector]
    hexagram_points: List[Vector]
    triangle_grid: List[Vector]
    hexagram_corner_points: Dict[int, List[Vector]]

    def __init__(self, radius: int):
        """
        Initialize a Hexagram object.

        More Info: https://en.wikipedia.org/wiki/Hexagram

        Args:
            radius: The distance from the center to an inner corner (inner radius) of the hexagram.

        Raises:
            Exception: If the provided radius is not greater than 0.
        """
        if radius <= 0:
            raise ValueError("Hexagram radius must be > 0")

        self.radius = radius

        # Define the six corner points of a hexagon using 2D basis vectors.
        self.hexagon_points: List[Vector] = [
            Vector(i, j) for i, j in [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        ]

        # Define edges of the hexagon by pairing adjacent corner points.
        self.hexagon_edges: List[Tuple[Vector, Vector]] = [
            (point, self.hexagon_points[(i + 1) % 6])
            for i, point in enumerate(self.hexagon_points)
        ]

        # Define a square lattice of points, whose dimensions align with the hexagram's inner radius.
        self.rhombus_grid: List[Vector] = [
            Vector(i, j)
            for i in range(self.radius + 1)
            for j in range(self.radius + 1)
        ]

        # Construct hexagram points by projecting a rhombus onto each hexagon edge.
        self.hexagram_points: List[Vector] = list({
            Vector(
                start.i * grid_point.i + end.i * grid_point.j,
                start.j * grid_point.i + end.j * grid_point.j,
            )
            for grid_point in self.rhombus_grid
            for (start, end) in self.hexagon_edges
        })

        # The triangle grid comprises points in the upper half of the rhombus grid.
        self.triangle_grid: List[Vector] = [
            Vector(i, j)
            for i in range(1, self.radius + 1)
            for j in range(self.radius - i + 1, self.radius + 1)
        ]

        # Map each corner of the hexagram to its respective points, based on projection onto each hexagon edge.
        self.hexagram_corner_points: Dict[int, List[Vector]] = {
            index: [
                Vector(
                    start.i * grid_point.i + end.i * grid_point.j,
                    start.j * grid_point.i + end.j * grid_point.j,
                )
                for grid_point in self.triangle_grid
            ]
            for index, (start, end) in enumerate(self.hexagon_edges)
        }

    def __eq__(self, other: 'Hexagram') -> bool:
        return self.radius == other.radius \
            and self.hexagon_points == other.hexagon_points \
            and self.hexagon_edges == other.hexagon_edges \
            and self.rhombus_grid == other.rhombus_grid \
            and self.hexagram_points == other.hexagram_points \
            and self.triangle_grid == other.triangle_grid \
            and self.hexagram_corner_points == other.hexagram_corner_points