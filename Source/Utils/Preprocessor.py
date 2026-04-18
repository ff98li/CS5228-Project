# DEPRECATED: Use Graph.Preprocessor instead

import numpy

from Utils.DataMatrix import DataMatrix

class Preprocessor:
    def __init__(self) -> None:
        self.m_nodes = []

    def push(self, node: TransformNode) -> None:
        self.m_nodes.append(node)

    def pop(self) -> None:
        if(len(self.m_nodes > 0)):
            self.m_nodes.pop(-1)

    def __call__(self, matrix: DataMatrix, shouldUndo: bool = False) -> None:
        if(not shouldUndo):
            for node in self.m_nodes:
                node.apply(matrix)
            return
        for node in reversed(self.m_nodes):
            node.undo(matrix)

# ================================================================================

class TransformNode:
    def __init__(self, targets: list):
        size = len(targets)
        self.m_targets = targets
        self.m_metadata = [None,] * size

    def apply(self, matrix: DataMatrix) -> None:
        for i, target in enumerate(self.m_targets):
            if(not target in matrix.m_colums):
                continue
            col = matrix.__getattr__(target)
            newVals = self.__fX__(col)
            self.m_metadata[i] = newVals
            print(f"Transformed: {target}")

    def undo(self, matrix: DataMatrix) -> None:
        for i, target in enumerate(self.m_targets):
            if(not target in matrix.m_colums):
                continue
            metadata = self.m_metadata[i]
            col = matrix.__getattr__(target)
            self.__fInvX__(col, metadata)
            print(f"Transform inverted: {target}")

# ================================================================================

class NormalizeNode(TransformNode):
    def __init__(self, colums: list) -> None:
        super().__init__(colums)

    def __fX__(self, target: numpy.array) -> tuple:
        minVal = numpy.min(target, axis = 0)
        maxVal = numpy.max(target, axis = 0)
        target -= minVal; target /= (maxVal - minVal)
        return (minVal, maxVal)

    def __fInvX__(self, target: numpy.array, metadata: tuple) -> None:
        maxVal, minVal = metadata
        target *= (maxVal - minVal)
        target += minVal

# ================================================================================

class StandardizeNode(TransformNode):
    def __init__(self, colums: list) -> None:
        super().__init__(colums)

    def __fX__(self, target: numpy.array) -> tuple:
        mean = numpy.mean(target, axis = 0)
        std = numpy.std(target, axis = 0)
        target -= mean; target /= std
        return (mean, std)

    def __fInvX__(self, target: numpy.array, metadata: tuple) -> None:
        mean, std = metadata
        target *= std; target += mean

# ================================================================================

class LabelEncoderNode(TransformNode):
    def __init__(self, colums: list, override: dict = None) -> None:
        super().__init__(colums)
        self.m_override = None

    def __fX__(self, target: numpy.array) -> np.array | None:
        if(self.m_override is None):
            unique  = numpy.unique(target)
            for i, val in enumerate(unique):
                target[target == val] = i
            return unique
        for key in self.m_override.keys():
            value = self.m_override[key]
            target[target == key] = value

    def __fInvX__(self, target: numpy.array, metadata: np.array) -> None:
        if(self.m_override is None):
            for i, val in enumerate(metadata):
                target[target == i] = val
            return None
        for key in self.m_override.keys():
            value = self.m_override[key]
            target[target == value] = key

# ================================================================================

class TypeConvNode:
    def __init__(self, dstType) -> None:
        self.m_dstType = dstType

    def apply(self, matrix: DataMatrix) -> None:
        result = matrix.m_values.astype(self.m_dstType)
        matrix.m_values = result
        print(f"Type casted to {self.m_dstType.__name__}")

    def undo(self, matrix: DataMatrix) -> None:
        print(f"Type casted to {object.__name__}")
        result = matrix.m_values.astype(object)
        matrix.m_values = result
