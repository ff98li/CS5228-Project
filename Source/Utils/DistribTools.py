import numpy

from Utils.DataMatrix import DataMatrix

class DistribGenerator:
    def __init__(self, resolution: int = 20) -> None:
        BIN_MIN = -5; BIN_MAX = 5
        delta = (BIN_MAX - BIN_MIN) / resolution
        self.m_bins = numpy.arange(BIN_MIN, BIN_MAX, delta)

    def getMarginal(self, source: DataMatrix, ID: str) -> numpy.array:
        result = source.__getattr__(ID)
        result, _ = numpy.histogram(result, bins = self.m_bins)
        result = result / numpy.sum(result)
        return result

    def getJoint(self, source: DataMatrix, IDs: list) -> numpy.array:
        result = source.getCols(IDs).m_values
        result, _ = numpy.histogramdd(result, [self.m_bins] * 2)
        result = result / numpy.sum(result)
        return result

    def getConditional(self, source: DataMatrix, target: str, condition: dict) -> numpy.array:
        result = source.clone()
        for ID in condition.keys():
            idx = result.m_colums.index(ID)
            mask = result.m_values[:, idx] == condition[ID]
            result.m_values = result.m_values[mask]
        return self.getMarginal(result, target)

def computeCovariance(x1: numpy.array, x2: numpy.array) -> float:
    mu1 = numpy.mean(x1)
    mu2 = numpy.mean(x2)
    var = (x1 - mu1) * (x2 - mu2)
    result = (numpy.sum(var)) / x1.shape[0]
    return result

def computeKLDivergence(p: numpy.array, q: numpy.array) -> float:
    p = p + 1e-6; p /= numpy.sum(p)
    q = q + 1e-6; q /= numpy.sum(q)
    result = p / q
    result = p * numpy.log(result)
    return numpy.sum(result)
