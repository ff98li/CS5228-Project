import numpy
import pandas
from matplotlib import pyplot

from Model.ConditionalBernoulli import ConditionalBernoulli
from Model.UnivariateDistrib import BernoulliDistrib
from Model.GaussianMixture import GaussianMixture
from Model.Optimizer import Optimizer
from Model.Sampler import Sampler

from Graph.DependencyGraph import DependencyGraph
from Graph.AttributeNode import AttributeNode
from Graph.Preprocessor import Preprocessor
from Graph.GraphProbe import GraphPrinter

from Utils.TransformLayer import TypeConvLayer
from Utils.TransformLayer import StandardizeLayer
from Utils.TransformLayer import LabelEncoderLayer


graph = DependencyGraph()

node = graph.attachNode(AttributeNode("State"), None, BernoulliDistrib())
node.addTransform(LabelEncoderLayer())
node.addTransform(TypeConvLayer(numpy.int32))

attribPairs = [
    ("Account length", "State"),
    ("Total day calls", "State"),
    ("Total eve calls", "State"),
    ("Total night calls", "State"),
    ("Total day charge", "State"),
    ("Total eve charge", "State"),
    ("Total night charge", "State")
]
for childID, parentID in attribPairs:
    node = graph.attachNode(AttributeNode(childID), parentID, GaussianMixture())
    node.addTransform(StandardizeLayer())
    node.addTransform(TypeConvLayer(numpy.float32))

attribPairs = [
    ("Voice mail plan", "Total day calls"),
    ("Voice mail plan", "Total eve calls"),
    ("International plan", "Total day calls"),
    ("International plan", "Total eve calls"),

    ("Churn", "International plan"),
    ("Churn", "Total day charge"),
]
for childID, parentID in attribPairs:
    node = graph.attachNode(AttributeNode(childID), parentID, ConditionalBernoulli())
    node.addTransform(LabelEncoderLayer({"Yes": 1, "No": 0}))
    node.addTransform(TypeConvLayer(numpy.int32))

graph.recurse(GraphPrinter(), "State")

# ================================================================================

IN_PATH  = "./Data/churn-bigml-80.csv"
dataset = pandas.read_csv(IN_PATH)
graph.recurse(Preprocessor(dataset), "State")

graph.recurse(Optimizer(dataset), "State")
graph.recurse(Sampler(), "State")
