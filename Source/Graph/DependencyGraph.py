from collections import deque

from Graph.AttributeNode import AttributeNode

class DependencyGraph:
    def __init__(self) -> None:
        self.m_nodes = {}
        self.m_distrib = {}

    def getNode(self, ID: str) -> AttributeNode | None:
        if(not ID in self.m_nodes):
            return None
        return self.m_nodes[ID]

    def getParents(self, ID: str) -> list:
        result = []
        for node in self.m_nodes.values():
            if(ID in node.m_childNodes):
                result.append(node.m_ID)
        return result

    def addDistrib(self, childID, parentID, distrib) -> None:
        ID = "" if parentID is None else parentID + "::"
        ID = f"{ID}{childID}"
        self.m_distrib[ID] = distrib

    def attachNode(self, node: AttributeNode, attachTo: str, distrib: object) -> AttributeNode:
        self.m_nodes[node.m_ID] = node
        if(not attachTo is None):
            parent = self.getNode(attachTo)
            parent.addChild(node)
        self.addDistrib(node.m_ID, attachTo, distrib)
        return node

    def recurse(self, probe: object, rootID: str) -> object:
        probe.m_graph = self
        queue = deque([(rootID, 1)])
        while queue:
            currentID, depth = queue.popleft()
            node = self.getNode(currentID)
            probe.m_depth = depth
            probe.evaluate(node)
            for ID, child in node.m_childNodes.items():
                queue.append((child.m_ID, depth + 1))
