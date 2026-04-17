import numpy

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

def computeJaccardIndex(p: numpy.array, q: numpy.array) -> float:
    I = numpy.sum(numpy.minimum(p, q))
    U = numpy.sum(numpy.maximum(p, q))
    return 1.0 - (I / U)
