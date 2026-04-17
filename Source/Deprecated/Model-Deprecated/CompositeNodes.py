import numpy
from matplotlib import pyplot

from Utils.DataMatrix import DataMatrix
from Utils.DistribGenerator import DistribGenerator

from Model.UnivariateNodes import GaussianNode

class ConditionalNode:
    # NOTE: Internally uses a gaussian mixture model

    def __init__(self, childID: str, parentID: str):
        self.m_childID = childID
        self.m_parentID = parentID
        self.m_weights = []
        self.m_nodes = []

    def __str__(self) -> str:
        result = f"ConditionalNode({self.m_childID}, {self.m_parentID}):"
        for i, node in enumerate(self.m_nodes):
            weight = self.m_weights[i]
            result += f"\n - {node} @ {weight}"
        return result

    def sample(self, size: int = 1, condition: dict) -> numpy.array:
        indices = numpy.arange(0, len(self.m_nodes))
        result = numpy.zeros((size, 1))
        components = numpy.zeros((size, 1), dtype = numpy.int32)
        for i in range(size):
            components[i] = numpy.random.choice(indices, p = self.m_weights)
            value = self.m_nodes[components[i].item()].sample()
            result[i] = value
        return result, components

    def fit(self, dataset: DataMatrix) -> float:
        parentNode = dataset.__getattr__(self.m_parentID)
        gen = DistribGenerator()
        for val in numpy.unique(parentNode):
            condition = {self.m_parentID: val}
            distrib, count = gen.getConditional(dataset, self.m_childID, condition)

            weight = count / parentNode.shape[0]
            node = GaussianNode()
            score = node.fit(distrib, True)
            if(score < 0.7):
                print(f"WARN: Node({self.m_childID} | {self.m_parentID} == {val}) Inaccurate")
            self.m_nodes.append(node)
            self.m_weights.append(weight)
