from Graph.AttributeNode import AttributeNode

class GraphProbe:
    def __init__(self) -> None:
        self.m_depth = 0
        self.m_graph = None

    def evaluate(self, node: AttributeNode) -> None:
        raise NotImplementedError

# NOTE: Sample probe implementation
class GraphPrinter(GraphProbe):
    def __init__(self) -> None:
        super().__init__()

    def evaluate(self, node: AttributeNode) -> None:
        indent = " " * self.m_depth * 2
        print(f"{indent}- {node.m_ID}")
