from abc import ABC, abstractmethod
from typing import List, Generic, TypeVar
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent
import torch

# input type T, output type S, output shape U
T = TypeVar('T')
S = TypeVar('S')
U = TypeVar('U')

class IEncoder(ABC, Generic[T, S, U]):
    """
    Generic interface for encoders. Encodes an object of type T into a tensor of type S.
    """

    @property
    @abstractmethod
    def input_type(self) -> type:
        """
        Returns the expected type of the input object for encoding.
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> U:
        """
        Defines the shape of the encoded torch.Tensor.

        Returns:
            Tuple[int, ...]: The shape of the output tensor, where each integer represents
                             the size of a specific dimension.
        """
        pass

    @abstractmethod
    def encode(self, obj: T) -> S:
        """
        Encodes an object of type T into a tensor of type S.

        Args:
            obj (T): The object to encode.

        Returns:
            S: The encoded representation of the object as a tensor.
        """
        pass

    def batch_encode(self, objs: List[T], batch_size: int) -> List[S]:
        """
        Encodes a list of objects in batches and parallelizes the process.

        Args:
            objs (List[T]): The list of objects to encode.
            batch_size (int): The number of objects to process in a single batch.

        Returns:
            List[S]: A list of encoded tensors.
        """
        encoded_objs = []

        batches = [objs[i:i + batch_size] for i in range(0, len(objs), batch_size)]

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._encode_batch, batch): batch for batch in batches}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(batches), desc="Encoding batches",
                               unit="batch"):
                try:
                    encoded_objs.extend(future.result())
                except Exception as ex:
                    print(f"Error encountered while encoding a batch: {ex}")

        return encoded_objs

    def _encode_batch(self, batch: List[T]) -> List[S]:
        """
        Encodes a single batch of objects.

        Args:
            batch (List[T]): A batch of objects to encode.

        Returns:
            List[S]: A list of encoded tensors.
        """
        return [self.encode(obj) for obj in batch]
