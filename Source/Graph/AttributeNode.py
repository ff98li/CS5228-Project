class AttributeNode:
    def __init__(self, ID: str) -> None:
        self.m_ID = ID
        self.m_transforms = []
        self.m_childNodes = {}

    def addTransform(self, operator: object) -> None:
        self.m_transforms.append(operator)

    def addChild(self, node: Attribute) -> AttributeNode:
        assert(not node.m_ID in self.m_childNodes)
        self.m_childNodes[node.m_ID] = node
        return node

    def getChild(self, ID: str):
        assert(ID in self.m_childNodes)
        return self.m_childNodes[ID]
