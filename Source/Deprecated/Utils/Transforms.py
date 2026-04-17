import numpy


class NormalizeOP:
    def __init__(self) -> None:
        self.m_min = 0.0
        self.m_max = 1e+5

    def apply(self, target: numpy.array) -> numpy.array:
        self.m_min = numpy.min(target)
        self.m_max = numpy.max(target)
        result = target - self.m_min
        result = result / (self.m_max - self.m_min)
        return result

    def invert(self, target: numpy.array) -> numpy.array:
        result = target * (self.m_max - self.m_min)
        result = result + self.m_min
        return result

class StandardizeOP:
    def __init__(self) -> None:
        self.m_mean = 0.0
        self.m_std = 1.0

    def apply(self, target: numpy.array) -> numpy.array:
        self.m_mean = numpy.mean(target)
        self.m_std  = numpy.std(target)
        result = target - self.m_mean
        result = result / self.m_std
        return result

    def invert(self, target: numpy.array, metadata: tuple) -> None:
        result = target * self.m_std
        result = result + self.m_mean
        return result

# ================================================================================

class LabelEncodeOP:
    def __init__(self, bindings: dict = None) -> None:
        self.m_bindings = None

    def apply(self, target: numpy.array) -> numpy.array:
        result = numpy.copy(target)
        if(self.m_bindings is None):
            unique  = numpy.unique(target)
            self.m_bindings = {value: i for i, value in enumerate(unique)}
        for key in self.m_bindings.keys():
            value = self.m_bindings[key]
            result[result == key] = value
        return result

    def invert(self, target: numpy.array) -> numpy.array:
        result = numpy.copy(target)
        for key in self.m_bindings.keys():
            value = self.m_bindings[key]
            result[result == value] = key
        return result

# ================================================================================

class TypeConvOP:
    def __init__(self, dstType) -> None:
        self.m_dstType = dstType
        self.m_srcType = object

    def apply(self, target: numpy.array) -> numpy.array:
        self.m_srcType = target.dtype
        result = target.astype(self.m_dstType)
        return result

    def invert(self, target: numpy.array) -> numpy.array:
        result = target.astype(self.m_srcType)
        return result
