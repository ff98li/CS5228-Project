import numpy
import pandas
import seaborn
from matplotlib import pyplot

from Encoder import LabelEncoder
from Filter import CorrFilter
from DistribProbe import *

SEPARATOR = "=" * 60
IN_PATH  = "./Data/churn-bigml-80.csv"

normVars = [
    "Account length",
    "Total day calls",
    "Total day charge",
    "Total eve calls",
    "Total eve charge",
    "Total night calls",
    "Total night charge"
]

catVars = [
    "State",
    "International plan",
    "Voice mail plan",
    "Churn"
]

# ============================================================

dataMatrix = pandas.read_csv(IN_PATH)
encoder = LabelEncoder()
dataMatrix = encoder(dataMatrix, "Churn"); encoder.m_bindings = None
dataMatrix = encoder(dataMatrix, "State"); encoder.m_bindings = None
dataMatrix = encoder(dataMatrix, "Voice mail plan"); encoder.m_bindings = None
dataMatrix = encoder(dataMatrix, "International plan"); encoder.m_bindings = None
print(dataMatrix.info())

dataMatrix.hist(bins = 10)
pyplot.show()
print(SEPARATOR)

# ============================================================

for var in normVars:
    mean = dataMatrix[var].mean()
    std = dataMatrix[var].std()
    dataMatrix[var] -= mean
    dataMatrix[var] /= std

for i in range(len(normVars)):
    for j in range(i + 1, len(normVars)):
        x = normVars[i]
        y = normVars[j]
        p = getMarginal(dataMatrix, x)
        q = getMarginal(dataMatrix, y)
        cov = computeCovariance(p, q)
        print(f"{x}, {y} :: {cov}")
        # distrib = getJoint(dataMatrix, [x, y])
        # pyplot.matshow(distrib, cmap = "hot")
        # pyplot.show()
print(SEPARATOR)

# ============================================================

def computeAvgDiv(x: str, y: str) -> float:
    pX = getMarginal(dataMatrix, x)
    yVals = dataMatrix[y].unique()
    result = 0.0
    for val in yVals:
        pXCY = getConditional(dataMatrix, [x,], {y: val})
        div = computeKLDiv(pX, pXCY)
        result += div / (len(yVals))
        # print(f"p({x}|{y} = {val})")
        # pyplot.plot(pX, label = "p(X)")
        # pyplot.plot(pXCY, label = "p(X|Y)")
        # pyplot.legend()
        # pyplot.show()
    return result

for x in normVars:
    for y in catVars:
        div = computeAvgDiv(x, y)
        marker = "<<" if div >= 0.1 else ""
        print(f"{x}, {y} :: {div}", marker)
print(SEPARATOR)

for i in range(len(catVars)):
    for j in range(i + 1, len(catVars)):
        x = catVars[i]
        y = catVars[j]
        div = computeAvgDiv(x, y)
        marker = "<<" if div >= 0.1 else ""
        print(f"{x}, {y} :: {div}", marker)
print(SEPARATOR)

# ============================================================

# for x in normVars:
#     for y in catVars:
#         print(f"p({x} | {y})")
#         for val in dataMatrix[y].unique():
#             pXCY = getConditional(dataMatrix, [x,], {y: val})
#             pyplot.plot(pXCY, label = f"p(X | Y = {val})", alpha = 0.7)
#         pyplot.legend()
#         pyplot.show()
# print(SEPARATOR)
