import re
import copy
import numpy
import pandas

class DataMatrix:
    def __init__(self, source: pandas.DataFrame) -> None:
        self.m_values = None
        self.m_colums = []
        if(not source is None):
            self.m_values = source.to_numpy()
            self.m_colums = [DataMatrix.createID(name) for name in source.columns]

    def __getattr__(self, ID: str) -> numpy.array:
        assert(ID in self.m_colums)
        idx = self.m_colums.index(ID)
        values = self.m_values[:, idx : idx + 1]
        return values

    def getCols(self, IDs: tuple):
        values = [self.__getattr__(i) for i in IDs]
        values = numpy.array(values).squeeze(-1)
        values = numpy.transpose(values)
        result = DataMatrix(None)
        result.m_values = values
        result.m_colums = IDs
        return result

    def dropCol(self, ID: str) -> None:
        assert(ID in self.m_colums)
        idx = self.m_colums.index(ID)
        self.m_colums.remove(ID)
        self.m_values = numpy.delete(self.m_values, idx, 1)
        print(f"Dropped column: {ID}")

    def shape(self, axis: int | None = None) -> int | tuple:
        if(axis is None):
            return tuple(self.m_values.shape)
        assert(0 <= axis < 2)
        return self.m_values.shape[axis]

    def clone(self):
        result = DataMatrix(None)
        result.m_colums = copy.deepcopy(self.m_colums)
        result.m_values = copy.deepcopy(self.m_values)
        return result

    @staticmethod
    def createID(source: str) -> str:
        result = re.sub(r'[^a-zA-Z]+', ' ', source)
        result = result.title().replace(' ', '')
        result = result[0].lower() + result[1:]
        return result
