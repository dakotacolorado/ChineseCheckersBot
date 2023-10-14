from src.chinese_checkers.geometry.Vector import Vector


class Position(Vector):
    def __init__(self, i: int, j: int):
        self.i = i
        self.j = j

