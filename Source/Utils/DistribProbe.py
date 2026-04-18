import copy
import numpy
import pandas

RESOLUTION = 15

def getMarginal(source: pandas.DataFrame, attrib: str) -> numpy.array:
    assert(attrib in source.columns)
    result = source[attrib].to_numpy()
    result, _ = numpy.histogram(result, bins = RESOLUTION)
    result = result / numpy.sum(result)
    return result

def getJoint(source: pandas.DataFrame, attrib: list) -> numpy.array:
    result = source[attrib].to_numpy()
    result, _ = numpy.histogramdd(result, bins = RESOLUTION)
    result = result / numpy.sum(result)
    return result

def getConditional(source: pandas.DataFrame, attrib: list, condition: dict) -> numpy.array:
    result = copy.deepcopy(source)
    for ID in condition.keys():
        value = condition[ID]
        result = result[result[ID] == value]
        result = result.drop(ID, axis = 1)
    if(len(attrib) > 1):
        return getJoint(result, attrib)
    return getMarginal(result, attrib[0])

def computeCovariance(x1: numpy.array, x2: numpy.array) -> float:
    mu1 = numpy.mean(x1)
    mu2 = numpy.mean(x2)
    var = (x1 - mu1) * (x2 - mu2)
    result = (numpy.sum(var)) / x1.shape[0]
    return result

def computeKLDiv(p: numpy.array, q: numpy.array) -> float:
    p = p + 1e-6; p /= numpy.sum(p)
    q = q + 1e-6; q /= numpy.sum(q)
    result = p / q
    result = p * numpy.log(result)
    return numpy.sum(result)
