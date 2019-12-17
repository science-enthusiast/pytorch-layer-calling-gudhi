import unittest 

import torch

from src.layer.pair_dist import PairDist

class TestLayerPairDist(unittest.TestCase):
    def test_init(self):
        m = PairDist()
     