import torch

class Tde(torch.nn.Module):
    def __init__(self, skip=1, delay=1, dimension=3):
        super(Tde, self).__init__()
        self.skip = skip
        self.delay = delay
        self.dimension = dimension

    def forward(self, timeSeries):
        if (len(timeSeries) == 1):
            return timeSeries
        else:
            numPts = (len(timeSeries) - (self.dimension - 1) * self.delay) // self.skip
            X = torch.zeros(numPts, self.dimension)
 
            for j in range(numPts):

                for k in range(self.dimension):
                    X[j,k] = timeSeries[j * self.skip + k * self.delay]
 
            return X 
