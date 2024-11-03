"""
This module defines interfaces for encoding components of the Chinese Checkers game.
These interfaces provide a consistent contract for implementing various encoding strategies,
ensuring flexibility and interoperability across different parts of the game engine.

Purpose:
--------
- To standardize the encoding of game states, moves, and experiences into formats suitable
  for machine learning models and other processing tasks.
- To facilitate the extension and customization of encoding strategies by providing a clear
  set of methods that each implementation must define.

Usage:
------
- Implement these interfaces when developing new encoding strategies, ensuring that all
  necessary methods are provided to meet the interface contract.
- Utilize the existing interfaces to maintain consistency and reliability in encoding
  implementations across the codebase.

Creating New Interfaces:
------------------------
- Consider creating a new interface if a distinct aspect of the game or encoding process
  is identified that requires separate abstraction and encapsulation.
- New interfaces should define a clear and minimal set of methods to support specific
  encoding needs and encourage modular design.

By adhering to these interfaces, developers can ensure that encoding components are
interchangeable and maintain a high level of code quality and adaptability.
"""

from .IChineseCheckersGameEncoder import IChineseCheckersGameEncoder
from .IMoveEncoder import IMoveEncoder
from .IExperienceEncoder import IExperienceEncoder
from .IEncoder import IEncoder

__all__ = [
    "IChineseCheckersGameEncoder",
    "IMoveEncoder",
    "IExperienceEncoder",
    "IEncoder",
]