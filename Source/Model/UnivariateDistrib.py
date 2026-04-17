import numpy

class BernoulliDistrib:
    def __init__(self) -> None:
        self.m_theta = None

    def fit(self, samples: numpy.array) -> None:
        _, self.m_theta = numpy.unique(samples, return_counts = True)
        self.m_theta = self.m_theta.astype(numpy.float32)
        self.m_theta /= numpy.sum(self.m_theta)

    def sample(self) -> int:
        indices = numpy.arange(0, len(self.m_theta))
        result = numpy.random.choice(indices, p = self.m_theta)
        return int(result.item())

    def invSample(self, value: int) -> float:
        return self.m_theta[value, None]

# ================================================================================

class GaussianDistrib:
    def __init__(self, mean: float = None, sigma: float = None) -> None:
        self.m_mean = mean
        self.m_sigma = sigma

    def fit(self, samples: numpy.array) -> None:
        if(self.m_mean is None):
            self.m_mean = numpy.mean(samples, axis = 0)
        if(self.m_sigma is None):
            self.m_sigma = numpy.std(samples, axis = 0)

    def sample(self) -> float:
        result = numpy.random.normal(self.m_mean, self.m_sigma)
        return result

    def invSample(self, sample: float) -> float:
        var = self.m_sigma ** 2.0
        exp = (sample - self.m_mean) ** 2.0
        exp = numpy.exp(-exp / (2.0 * var))
        result = exp / ((2.0 * numpy.pi * var) ** 0.5)
        return result.item()
