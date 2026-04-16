import numpy

from Utils.DataMatrix import DataMatrix

class DistribGenerator:
    s_range = (-5, 5)

    def __init__(self, resolution: int = 20) -> None:
        delta = (self.s_range[1] - self.s_range[0]) / resolution
        self.m_bins = numpy.arange(self.s_range[0], self.s_range[1], delta)

    def __call__(self, values: numpy.array) -> numpy.array:
        result, _ = numpy.histogram(values, bins = self.m_bins)
        result = result / numpy.sum(result)
        return result

    def getMarginal(self, source: DataMatrix, ID: str) -> numpy.array:
        return self(source.__getattr__(ID))

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
