import numpy

from Model.EvalMetrics import computeJaccardIndex

from Utils.DistribGenerator import DistribGenerator


class GaussianDistrib:
    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.m_mean = mean
        self.m_sigma = std

    def __str__(self) -> str:
            return f"GaussianNode({self.m_mean:.2f}, {self.m_sigma:.2f})"

    def sample(self, size: int = 1) -> numpy.array:
        return numpy.random.normal(self.m_mean, self.m_sigma, (size, 1))

    def invSample(self, sample: numpy.array) -> numpy.array:
        var = self.m_sigma ** 2.0
        exp = (sample - self.m_mean) ** 2.0
        exp = numpy.exp(-exp / (2.0 * var))
        result = exp / ((2.0 * numpy.pi * var) ** 0.5)
        return result

    def fit(self, values: numpy.array, isStandardized: bool = False) -> float:
        assert(values.shape[1] == 1)

        if(not isStandardized):
            self.m_mean = numpy.mean(values, axis = 0)
            self.m_sigma = numpy.std(values, axis = 0)

        gen = DistribGenerator()
        src = gen(values)
        dst = gen(self.sample(len(values) * 2))
        return computeJaccardIndex(src, dst)
