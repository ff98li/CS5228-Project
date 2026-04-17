import numpy
import pandas
from matplotlib import pyplot

from Utils.DataMatrix import DataMatrix

from Utils.Preprocessor import Preprocessor
from Utils.Preprocessor import NormalizeNode
from Utils.Preprocessor import StandardizeNode
from Utils.Preprocessor import LabelEncoderNode
from Utils.Preprocessor import TypeConvNode

from Utils.DistribGenerator import DistribGenerator

from Model.EvalMetrics import computeCovariance
from Model.EvalMetrics import computeKLDivergence

SEPARATOR = "=" * 60
IN_PATH  = "./Data/churn-bigml-80.csv"

normVars = [
    "accountLength",
    "totalDayCalls",
    "totalDayCharge",
    "totalEveCalls",
    "totalEveCharge",
    "totalNightCalls",
    "totalNightCharge"
]

catVars = [
    "state",
    "internationalPlan",
    "voiceMailPlan",
    "churn"
]

# ============================================================

dataset = pandas.read_csv(IN_PATH)
dataset = DataMatrix(dataset)
dataset.dropCol("areaCode")

pproc = Preprocessor()
pproc.push(LabelEncoderNode(["state"]))
pproc.push(LabelEncoderNode(["internationalPlan", "voiceMailPlan"], {"Yes": 1, "No": 0}))
pproc.push(TypeConvNode(numpy.float32))
pproc.push(StandardizeNode(["accountLength"]))
pproc.push(StandardizeNode(["totalDayCalls", "totalDayCharge"]))
pproc.push(StandardizeNode(["totalEveCalls", "totalEveCharge"]))
pproc.push(StandardizeNode(["totalNightCalls", "totalNightCharge"]))
pproc.push(StandardizeNode(["totalIntlCalls", "totalIntlCharge"]))
pproc(dataset)

# ============================================================

generator = DistribGenerator()

for i in range(len(normVars)):
    for j in range(i + 1, len(normVars)):
        x = normVars[i]
        y = normVars[j]
        p, _ = generator.getMarginal(dataset, x)
        q, _ = generator.getMarginal(dataset, y)
        cov = computeCovariance(p, q)
        print(f"{x}, {y} :: {cov}")
        # distrib, _ = generator.getJoint(dataset, [x, y])
        # pyplot.matshow(distrib, cmap = "hot")
        # pyplot.show()
print(SEPARATOR)

# ============================================================

def computeAvgDiv(x: str, y: str) -> float:
    idx = dataset.m_colums.index(y)
    pX, _ = generator.getMarginal(dataset, x)
    yVals = numpy.unique(dataset.m_values[:, idx])
    result = 0.0
    for val in yVals:
        pXCY, _ = generator.getConditional(dataset, x, {y: val})
        div = computeKLDivergence(pX, pXCY)
        result += div / len(yVals)
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

# ============================================================

for i in range(len(catVars)):
    for j in range(i + 1, len(catVars)):
        x = catVars[i]
        y = catVars[j]
        div = computeAvgDiv(x, y)
        marker = "<<" if div >= 0.01 else ""
        print(f"{x}, {y} :: {div}", marker)
print(SEPARATOR)
