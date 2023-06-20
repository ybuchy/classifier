import numpy as np
from typing import Union, Optional

class Tensor:
    # cur IDEA: if you have a tensor and do some calculation, then a new tensor will be created with a backward_function that has the parent tensors and how the gradient of the function is (~> NO in-place operations for Tensors with requires_grad=True)
    def __init__(data: Union[int, float, list, np.ndarray], requires_grad: Optional[bool]=None):
        if isinstance(data, (int, float)):
            self.data = np.array([data])

        elif isinstance(data, list):
            self.data = np.array(data)

        if requires_grad:
            self.grad = np.zeros(self.data.shape)


    # TODO
    @property
    def grad_tracked(self):
        pass

    def __add__(self, rhs: Tensor):
        requires_grad = self.requires_grad or rhs.requires_grad
        s = Tensor(self.data + rhs, requires_grad=self.requires_grad)
        # TODO now give s self as parent and some add function to know how to calc grad?
        # TODO check Tensor requires grad

    def sum(self):
        # This will be for grad to work (why needed?)
        pass

    def backward(self, grad=None):
        if 
        if self.back_fn is None: raise AttributeError("back function missing")
        if not self.back_fn.requires_grad: return

        for parent in back_fn.parents:
            if not parent.requires_grad: continue
            # example: linear, then the units are one tensor and weights are another tensor. optimizer will track weight tensor so this will add grad to weights but only forward the gradient through the layer
            if parent.grad_tracked:
                parent.grad += back_fn.backward(grad)
            parent.backward(grad)
        pass
        

class Function:
    # TODO
    def __init__(self, *tensors: Sequence[Tensor]):
        self.parents = tensors
        self.requires_grad = any(tensor.requires_grad for tensor in tensors)

    def forward(self):
        raise NotImplementedError(f"forward function of {type(self)} not implemented")

    def backward(self):
        raise NotImplementedError(f"backward function of {type(self)} not implemented")

    def __call__(self):
        pass
