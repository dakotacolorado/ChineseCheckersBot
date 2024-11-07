from typing import List


from .Vector import Vector

class Centroid(Vector):

    @staticmethod
    def from_vectors(vectors: List[Vector]) -> 'Centroid':
        if not vectors:
            raise ValueError("Cannot compute centroid of empty vector list")

        avg_i = sum(vector.i for vector in vectors) / len(vectors)
        avg_j = sum(vector.j for vector in vectors) / len(vectors)

        return Centroid(avg_i, avg_j)
