import numpy

from Utils.DataMatrix import DataMatrix

class DistribGenerator:
    s_range = (-5, 5)
    s_resolution = 20

    def __init__(self) -> None:
        delta = (self.s_range[1] - self.s_range[0]) / self.s_resolution
        self.m_bins = numpy.arange(self.s_range[0], self.s_range[1], delta)

    def __call__(self, values: numpy.array) -> numpy.array:
        result, _ = numpy.histogram(values, bins = self.m_bins, density = True)
        result = numpy.expand_dims(result, -1)
        return result

    def getMarginal(self, source: DataMatrix, ID: str) -> tuple:
        values = source.__getattr__(ID)
        return self(values), values.shape[0]

    def getJoint(self, source: DataMatrix, IDs: list) -> tuple:
        result = source.getCols(IDs).m_values
        result, _ = numpy.histogramdd(result, [self.m_bins] * 2, density = True)
        return result, result.shape[0]

    def getConditional(self, source: DataMatrix, target: str, condition: dict) -> tuple:
        result = source.clone()
        for ID in condition.keys():
            idx = result.m_colums.index(ID)
            mask = result.m_values[:, idx] == condition[ID]
            result.m_values = result.m_values[mask]
        return self.getMarginal(result, target)
