import unittest 

import torch
import numpy as np

from src.layer.rips import Rips

class TestLayerRips(unittest.TestCase):
    def test_init(self):
        m = Rips(1,10)
    def test_forward(self):
        m = Rips(1,10)
        x = torch.Tensor([[0,1,2],[1,0,3],[2,3,0]])
        y = m(x)
