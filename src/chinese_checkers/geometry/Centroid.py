from typing import List
from .Vector import Vector


class Centroid:
    def __init__(self, position: Vector):
        self.position = position

    @staticmethod
    def from_vectors(vectors: List[Vector]) -> 'Centroid':
        if not vectors:
            # Return centroid at origin if no vectors provided
            return Centroid(Vector(0.0, 0.0))

        # Calculate average i and j components
        avg_i = sum(vector.i for vector in vectors) / len(vectors)
        avg_j = sum(vector.j for vector in vectors) / len(vectors)

        # Create centroid at the computed average position
        return Centroid(Vector(avg_i, avg_j))

    def __repr__(self):
        return f"Centroid(position={self.position})"
