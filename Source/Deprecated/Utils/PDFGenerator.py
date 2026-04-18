import numpy
import pandas

class PDFGenerator:
    s_range = (-5, 5)
    s_resolution = 20

    def __init__(self) -> None:
        delta = (self.s_range[1] - self.s_range[0]) / self.s_resolution
        self.m_bins = numpy.arange(self.s_range[0], self.s_range[1], delta)
        self.m_bins += delta / 2.0

    def toPDF(self, values: numpy.array) -> numpy.array:
        result, _ = numpy.histogram(values, bins = self.m_bins)
        result = result / numpy.sum(result)
        return numpy.expand_dims(result, -1)

    def getMarginal(self, values: pandas.DataFrame, ID: str) -> numpy.array:
        values = values[ID].to_numpy()
        return self.toPDF(values)

    def getJoint(self, source: pandas.DataFrame, IDs: list) -> tuple:
        result = source[IDs].to_numpy()
        result, _ = numpy.histogramdd(result, [self.m_bins] * 2)
        result = result / numpy.sum(result)
        return result
