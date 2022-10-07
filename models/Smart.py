## this model is only for test purpose

import torch.nn as nn
import torch
import numpy as np


class Smart(nn.Module):

    def __init__(self,

        Nprod_Veclen,

        **kwargs
    ):
        super().__init__()

        self.Nprod_Veclen = Nprod_Veclen


    def forward(self, IN, PROB):

        return PROB






        

