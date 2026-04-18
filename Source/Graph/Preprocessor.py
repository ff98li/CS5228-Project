from Graph.GraphProbe import GraphProbe

class Preprocessor(GraphProbe):
    def __init__(self, target: pandas.DataFrame, invert: bool = False) -> None:
        super().__init__()
        self.m_skip = []
        self.m_target = target
        self.m_invert = invert

    def evaluate(self, node: AttributeNode) -> None:
        if(node.m_ID in self.m_skip):
            return
        transforms = node.m_transforms
        if(self.m_invert):
            transforms = reversed(transforms)
        for layer in transforms:
            values = self.m_target[node.m_ID].to_numpy()
            result = layer.invert(values) if self.m_invert else layer.apply(values)
            self.m_target[node.m_ID] = result
        self.m_skip.append(node.m_ID)
