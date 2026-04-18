from Graph.GraphProbe import GraphProbe
from Graph.AttributeNode import AttributeNode

class Sampler(GraphProbe):
    def __init__(self) -> None:
        super().__init__()
        self.m_sample = {}

    def evaluate(self, node: AttributeNode) -> None:
        if(node.m_ID in self.m_graph.m_distrib):
            distrib = self.m_graph.m_distrib[node.m_ID]
            self.m_sample[node.m_ID] = distrib.sample()

        for parentID in self.m_graph.getParents(node.m_ID):
            distID = f"{parentID}::{node.m_ID}"
            distrib = self.m_graph.m_distrib[distID]
            parentVal = self.m_sample[parentID]
            sample, _ = distrib.sample(parentVal)
            self.m_sample[node.m_ID] = sample

        print(self.m_sample)
