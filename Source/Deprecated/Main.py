import numpy
import pandas
from matplotlib import pyplot

from Model.Attribute import Attribute
from Model.UnivariateDistrib import BernouliDistrib
from Model.UnivariateDistrib import GaussianDistrib

from Utils.Transforms import TypeConvOP
from Utils.Transforms import LabelEncodeOP

IN_PATH  = "./Data/churn-bigml-80.csv"
dataset = pandas.read_csv(IN_PATH)

# ================================================================================

node = Attribute("State", BernouliDistrib())
node.addTransform(LabelEncodeOP())
node.addTransform(TypeConvOP(numpy.int32))
node.update(dataset)

sample = node.get()
prob = node.getProb(sample)
print(sample, prob)

# ================================================================================

node = Attribute("Account length", GaussianDistrib())
node.addTransform(TypeConvOP(numpy.float32))
node.update(dataset)

sample = node.get()
prob = node.getProb(sample)
print(sample, prob)

# ================================================================================

print(dataset[["State", "Account length"]])
