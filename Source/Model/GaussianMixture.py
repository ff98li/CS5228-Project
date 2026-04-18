import numpy

from Model.UnivariateDistrib import GaussianDistrib
from Model.UnivariateDistrib import BernoulliDistrib

class GaussianMixture:
    def __init__(self) -> None:
        self.m_weights = []
        self.m_sources = []

    def fit(self, samples: numpy.array) -> None:
        components = samples[:, 0]
        self.m_weights = BernoulliDistrib()
        self.m_weights.fit(components)

        values = samples[:, 1]
        for index in numpy.unique(components):
            distrib = values[components == index]
            newGaussian = GaussianDistrib()
            newGaussian.fit(distrib)
            self.m_sources.append(newGaussian)

    def sample(self, component: int = None) -> tuple:
        indices = numpy.arange(0, len(self.m_sources))
        resultC = component
        if(resultC is None):
            resultC = self.m_weights.sample()
        resultV = self.m_sources[resultC].sample()
        return (resultV, resultC)

    def invSample(self, sample: tuple) -> float:
        value, component = sample
        result = self.m_sources[component].invSample(value)
        return result
