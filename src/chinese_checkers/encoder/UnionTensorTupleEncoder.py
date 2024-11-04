import logging
from typing import List, Tuple, Type
import torch
from .ITensorTupleEncoder import ITensorTupleEncoder


class UnionTensorTupleEncoder(ITensorTupleEncoder):
    """
    Unions multiple ITensorTupleEncoder instances by concatenating their outputs.
    All encoders must have matching tuple sizes for successful union, ensuring compatibility
    across tensor shapes. If tuple sizes differ, an error is raised detailing each encoder's
    type and respective tuple length.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, encoders: List[ITensorTupleEncoder], input_type: Type):
        """
        Initializes the UnionTensorTupleEncoder with a list of encoders and an input type.

        Args:
            encoders (List[ITensorTupleEncoder]): List of encoders to be unioned.
            input_type (Type): Expected input type for the encoder.

        Raises:
            ValueError: If encoders have mismatched tuple sizes, an error is raised with details.
        """
        self.encoders = encoders
        self._input_type = input_type

        # Validate that all encoders have matching tuple sizes
        tuple_sizes = [len(encoder.shape) for encoder in encoders]
        if len(set(tuple_sizes)) > 1:
            encoder_details = ', '.join(
                f"{type(encoder).__name__} (tuple length: {len(encoder.shape)})"
                for encoder in encoders
            )
            self.logger.error(
                f"Mismatched tuple sizes in encoders. Expected all to match, but found: {encoder_details}"
            )
            raise ValueError(
                f"All encoders must have matching tuple sizes, but received encoders with mismatched sizes: {encoder_details}"
            )
        self.logger.debug("UnionTensorTupleEncoder initialized successfully with matching tuple sizes.")

    @property
    def input_type(self) -> Type:
        """
        Returns the expected type of the input object for encoding.

        Returns:
            Type: The type of input this encoder expects.
        """
        return self._input_type

    @property
    def shape(self) -> Tuple[Tuple[int, ...], ...]:
        """
        Defines the concatenated shape of all encoders in the union.

        Returns:
            Tuple[Tuple[int, ...], ...]: The combined shape of the output tuple, where
                                         each inner tuple represents the shape of a concatenated tensor.
        """
        concatenated_shape = tuple(
            tuple(sum(encoder.shape[i][dim] for encoder in self.encoders) for dim in
                  range(len(self.encoders[0].shape[i])))
            for i in range(len(self.encoders[0].shape))
        )
        self.logger.debug(f"Union shape calculated as: {concatenated_shape}")
        return concatenated_shape

    def encode(self, data) -> Tuple[torch.Tensor, ...]:
        """
        Encodes the data by concatenating the outputs from each encoder in the union.

        Args:
            data: The data to be encoded by each ITensorTupleEncoder.

        Returns:
            Tuple[torch.Tensor, ...]: The concatenated tuple of tensors from each encoder.
        """
        if not isinstance(data, self._input_type):
            self.logger.error(f"Input data type mismatch. Expected {self._input_type}, got {type(data)}.")
            raise TypeError(f"Expected input type {self._input_type}, but received {type(data)}.")

        encoded_outputs = [encoder.encode(data) for encoder in self.encoders]
        self.logger.debug("Individual encoders have successfully encoded the input data.")

        # Concatenate tensors in the same positions across all encoded outputs
        concatenated_output = tuple(
            torch.cat([encoded_output[i] for encoded_output in encoded_outputs], dim=-1)
            for i in range(len(encoded_outputs[0]))
        )

        self.logger.debug(f"Final concatenated output shape: {[tensor.shape for tensor in concatenated_output]}")
        return concatenated_output
