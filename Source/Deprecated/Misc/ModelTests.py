import numpy
import pandas
from matplotlib import pyplot

from Utils.DataMatrix import DataMatrix
from Utils.DistribGenerator import DistribGenerator

from Utils.Preprocessor import Preprocessor
from Utils.Preprocessor import NormalizeNode
from Utils.Preprocessor import StandardizeNode
from Utils.Preprocessor import LabelEncoderNode
from Utils.Preprocessor import TypeConvNode

from Model.UnivariateNodes import GaussianNode
from Model.CompositeNodes import ConditionalNode
from Model.EvalMetrics import computeJaccardIndex

IN_PATH  = "./Data/churn-bigml-80.csv"
dataset = pandas.read_csv(IN_PATH)
dataset = DataMatrix(dataset)
dataset.dropCol("areaCode")

node = ModelNode("totalDayCalls",
    typeID = typeID.normal,
    distrib = UnivariateDistrib()
    preproc = [],
)



pproc = Preprocessor()
pproc.push(LabelEncoderNode(["state"]))
pproc.push(LabelEncoderNode(["internationalPlan", "voiceMailPlan"], {"Yes": 1, "No": 0}))
pproc.push(TypeConvNode(numpy.float32))
pproc.push(StandardizeNode(["totalDayCalls", "totalDayCharge"]))
pproc.push(StandardizeNode(["totalEveCalls", "totalEveCharge"]))
pproc.push(StandardizeNode(["totalNightCalls", "totalNightCharge"]))
pproc.push(StandardizeNode(["totalIntlCalls", "totalIntlCharge"]))
pproc(dataset)

node = ConditionalNode("totalIntlCharge", "state")
node.fit(dataset)
print(node)

gen = DistribGenerator()
samples, _ = node.sample(dataset.shape(0))
estimate = gen(samples)
source, _ = gen.getMarginal(dataset, "totalDayCalls")

pyplot.plot(source, label = "souce")
pyplot.plot(estimate, label = "estimate")
pyplot.legend()
pyplot.show()
