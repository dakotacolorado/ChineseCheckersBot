from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Generic, Tuple, TypeVar
import torch
from tqdm import tqdm
import concurrent

T = TypeVar('T')

class IEncoder(ABC, Generic[T]):
    """
    Generic interface for encoders. Encodes an object of type T into a torch.Tensor.
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
    def shape(self) -> Tuple[int, ...]:
        """
        Returns the shape of the output tensor from the encoding.

        Returns:
            Tuple[int, ...]: Shape of the encoded torch.Tensor.
        """
        pass

    @abstractmethod
    def encode(self, obj: T) -> torch.Tensor:
        """
        Encodes an object of type T into a torch.Tensor.

        Args:
            obj (T): The object to encode.

        Returns:
            torch.Tensor: The encoded representation of the object.
        """
        pass

    def batch_encode(self, objs: List[T], batch_size: int) -> List[torch.Tensor]:
        """
        Encodes a list of objects in batches and parallelizes the process.

        Args:
            objs (List[T]): The list of objects to encode.
            batch_size (int): The number of objects to process in a single batch.

        Returns:
            List[torch.Tensor]: A list of encoded tensors.
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

    def _encode_batch(self, batch: List[T]) -> List[torch.Tensor]:
        """
        Encodes a single batch of objects.

        Args:
            batch (List[T]): A batch of objects to encode.

        Returns:
            List[torch.Tensor]: A list of encoded tensors.
        """
        return [self.encode(obj) for obj in batch]
