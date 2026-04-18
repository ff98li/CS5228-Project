import numpy

class NormalizeLayer:
    def __init__(self) -> None:
        self.m_min = 0.0
        self.m_max = 1e+6

    def apply(self, values: numpy.array) -> numpy.array:
        self.m_min = numpy.min(values, axis = 0)
        self.m_max = numpy.max(values, axis = 0)
        result = values - self.m_min
        result = result / (self.m_max - self.m_min)
        return result

    def invert(self, values: numpy.array) -> numpy.array:
        result = values * (self.m_max - self.m_min)
        result = result + self.m_min
        return result

# ================================================================================

class StandardizeLayer:
    def __init__(self) -> None:
        self.m_mean = 0.0
        self.m_std = 1.0

    def apply(self, values: numpy.array) -> numpy.array:
        self.m_mean = numpy.mean(values, axis = 0)
        self.m_std  = numpy.std(values, axis = 0)
        return (values - self.m_mean) / self.m_std

    def invert(self, values: numpy.array) -> numpy.array:
        return (values * self.m_std) + self.m_mean

# ================================================================================

class LabelEncoderLayer:
    def __init__(self, bindings: dict = None) -> numpy.array:
        self.m_bindings = bindings

    def apply(self, values: numpy.array) -> np.array:
        values = values.copy()
        if(self.m_bindings is None):
            unique  = numpy.unique(values)
            self.m_bindings = {ID: i for i, ID in enumerate(unique)}
        for ID, index in self.m_bindings.items():
            values[values == ID] = index
        return values

    def invert(self, values: numpy.array) -> numpy.array:
        values = values.copy()
        for ID, index in self.m_bindings.items():
            values[values == index] = ID
        return values

# ================================================================================

class TypeConvLayer:
    def __init__(self, dstType) -> None:
        self.m_dstType = dstType

    def apply(self, values: numpy.array) -> numpy.array:
        return values.astype(self.m_dstType)

    def invert(self, values: numpy.array) -> numpy.array:
        return values.astype(object)
