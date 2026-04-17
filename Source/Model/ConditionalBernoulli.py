import numpy

from Model.UnivariateDistrib import BernoulliDistrib
from Model.GaussianMixture import GaussianMixture

class ConditionalBernoulli:
    def __init__(self) -> None:
        self.m_prior = BernoulliDistrib()
        self.m_likelihood = GaussianMixture()

    def fit(self, samples: numpy.array) -> None:
        self.m_prior.fit(samples[:, 1])
        self.m_likelihood.fit(samples[:, [1, 0]])

    def sample(self, sampleV: float) -> tuple:
        prob = self.invSample((sampleV, None))
        result = numpy.random.choice(len(prob), p = prob)
        return (result, sampleV)

    def invSample(self, sample: tuple) -> numpy.array | float:
        value, component = sample
        jointProb = numpy.zeros_like(self.m_prior.m_theta)
        for k in range(jointProb.shape[0]):
            pY   = self.m_prior.invSample(k)
            pXCY = self.m_likelihood.invSample((value, k))
            jointProb[k] = (pY * pXCY).item()
        jointProb = jointProb / numpy.sum(jointProb)
        result = jointProb if component is None else jointProb[component]
        return result
