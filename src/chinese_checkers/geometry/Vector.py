class Vector:
    def __init__(self, i: float, j: float):
        self.i = i
        self.j = j

    def __eq__(self, other: 'Vector') -> bool:
        if not isinstance(other, Vector):
            return False
        return self.i == other.i and self.j == other.j

    def __hash__(self):
        return hash((self.i, self.j))

    def __repr__(self):
        return f"({self.i}, {self.j})"

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.i + other.i, self.j + other.j)

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.i - other.i, self.j - other.j)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector(self.i * other, self.j * other)
        elif isinstance(other, Vector):
            return Vector(self.i * other.i, self.j * other.j)
        else:
            raise TypeError(f"Cannot multiply Vector by {type(other)}")

    def distance(self, other: 'Vector') -> float:
        return ((self.i - other.i) ** 2 + (self.j - other.j) ** 2) ** 0.5
