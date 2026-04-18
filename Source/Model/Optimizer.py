from Graph.GraphProbe import GraphProbe
from Graph.AttributeNode import AttributeNode

class Optimizer(GraphProbe):
    def __init__(self, dataset: pandas.DataFrame) -> None:
        super().__init__()
        self.m_dataset = dataset

    def evaluate(self, node: AttributeNode) -> None:
        if(node.m_ID in self.m_graph.m_distrib):
            values = self.m_dataset[node.m_ID]
            distrib = self.m_graph.m_distrib[node.m_ID]
            distrib.fit(values.to_numpy())
            print(f"Optimized node: {node.m_ID}")

        for parentID in self.m_graph.getParents(node.m_ID):
            values = self.m_dataset[[parentID, node.m_ID]]
            distID = f"{parentID}::{node.m_ID}"
            distrib = self.m_graph.m_distrib[distID]
            distrib.fit(values.to_numpy())
            print(f"Optimized node: {distID}")
