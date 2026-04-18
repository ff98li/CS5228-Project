import pandas
import seaborn
from matplotlib import pyplot

from Utils.Encoder import VecEncoder
from Utils.Encoder import LabelEncoder
from Utils.Filter import CorrFilter

OUT_PATH = "./Results/EDA"
IN_PATH  = "./Data/churn-bigml-80.csv"
dataset = pandas.read_csv(IN_PATH)

# NOTE: Label distribution
labels = dataset["Churn"]
x = ["True", "False"]
y = [labels.sum(), len(labels) - labels.sum()]
print("Label Ratio:", y[0] / y[1])
pyplot.title("Label distribution")
pyplot.bar(x = x, height = y)
pyplot.savefig(f"{OUT_PATH}/LabelDistrib.png")
pyplot.close()

# NOTE: Encoding / Filtering
encoder = VecEncoder()
dataset = encoder(dataset, "State")
encoder = LabelEncoder({"Yes": 1, "No": 0})
dataset = encoder(dataset, "Voice mail plan")
dataset = encoder(dataset, "International plan")
filter = CorrFilter()
dataset = filter(dataset)

# NOTE: Correlation
corr = abs(dataset.corr())
pyplot.matshow(corr)
pyplot.savefig(f"{OUT_PATH}/Correl-Postfilter.png")

# NOTE: Plots
fig, ax = pyplot.subplots(1, 1, figsize=(18, 18))
dataset.hist(ax = ax, bins = 100)
pyplot.savefig(f"{OUT_PATH}/Distrib.png")
pyplot.close()
