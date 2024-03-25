"""
This file is used to transform the range of the tensor from one range to another.
"""


class RangeTransform:

    def __init__(self, in_range, out_range):
        """
        This is the constructor of the class.
        :param in_range: tuple, input range
        :param out_range: tuple, output range
        """
        self.in_range = in_range
        self.out_range = out_range

    def __call__(self, tensor):
        """
        This method is called when the transform is applied to the tensor.
        :param tensor: torch.Tensor, tensor to be transformed
        :return: torch.Tensor, transformed tensor
        """
        tensor = tensor - self.in_range[0]
        tensor = tensor / (self.in_range[1] - self.in_range[0])
        tensor = tensor * (self.out_range[1] - self.out_range[0])
        tensor = tensor + self.out_range[0]

        # make the tensor float32
        tensor = tensor.float()

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '()'
