import torch
from torch import nn


class MCDualMixin:
    """Monte Carlo mixin
    This mixin provide a method `sample` to sample from defined model
    Use this Mixin by inheriting this class
    Assuming that model returns a tuple of 2 tensors"""

    #def get_output_shape(self, *args):
    #    "Override this to get output dimensions."
    #    raise NotImplementedError("Need to define output shape")
    
    def sample(self, T:int, batch_size, input_value, input_time):
        # Construct empty outputs
        #shape_m, shape_v = self.get_output_shape(*args)
        M, V = torch.empty(T, batch_size, 1), torch.empty(T, batch_size, 1)
        
        for t in range(T):
            M[t], V[t], _ = self(input_value, input_time)
        
        return M, V
    


class MCSingleMixin:
    """Monte Carlo mixin
    This mixin provide a method `sample` to sample from defined model
    Use this Mixin by inheriting this class
    Assuming that model returns a single tensors"""

    
    def sample(self, T:int, batch_size, input_value, input_time):
        # Construct empty outputs
        #shape_m = self.get_output_shape(*args)
        M = torch.empty(T, batch_size, 1)
        
        for t in range(T):
            M[t] = self(input_value, input_time)
        
        return M