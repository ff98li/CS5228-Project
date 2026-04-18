from __future__ import annotations
import os
import numpy
import pandas
from scipy import stats
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from Utils.Encoder import VecEncoder, LabelEncoder
from Utils.Filter import CorrFilter

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH     = os.path.join(ROOT_DIR, "Data", "churn-bigml-80.csv")
OUT_DIR     = os.path.join(ROOT_DIR, "Results", "Distributions")
SUMMARY_PATH = os.path.join(ROOT_DIR, "Results", "Distributions", "step1_summary.csv")

SCALERS = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler":  MinMaxScaler(),
    "RobustScaler":  RobustScaler(),
}


class FeatureAnalyzer:
    def __init__(self) -> None:
        self.m_results = []

    def _recommend_scaler(self, skewness: float) -> str:
        absSkew = abs(skewness)
        if absSkew > 1.0:
            return "RobustScaler"
        elif absSkew > 0.5:
            return "MinMaxScaler"
        else:
            return "StandardScaler"

    def __call__(self, features: pandas.DataFrame, target: pandas.Series) -> pandas.DataFrame:
        self.m_results = []
        for col in features.columns:
            values = features[col].values
            skewness  = stats.skew(values)
            kurtosis  = stats.kurtosis(values)  # excess / Fisher
            shapiroW, shapiroP = stats.shapiro(values)
            isNormal = shapiroP > 0.05
            recommended = self._recommend_scaler(skewness)
            self.m_results.append({
                "feature":            col,
                "skewness":           skewness,
                "kurtosis":           kurtosis,
                "shapiro_pvalue":     shapiroP,
                "is_normal":          isNormal,
                "recommended_scaler": recommended,
            })
        return pandas.DataFrame(self.m_results)


class DistributionPlotter:
    def __init__(self, outDir: str) -> None:
        self.m_outDir = outDir

    def _plot_hist_kde(self, name: str, values: numpy.ndarray) -> None:
        path = os.path.join(self.m_outDir, f"{name}_hist_kde.png")
        pyplot.figure()
        pyplot.hist(values, bins = 30, density = True, alpha = 0.6, color = "steelblue", edgecolor = "white")
        kdeX = numpy.linspace(values.min(), values.max(), 200)
        kde = stats.gaussian_kde(values)
        pyplot.plot(kdeX, kde(kdeX), color = "darkred", linewidth = 1.5)
        pyplot.title(f"Distribution of {name}")
        pyplot.xlabel(name)
        pyplot.ylabel("Density")
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()

    def _plot_qq(self, name: str, values: numpy.ndarray) -> None:
        path = os.path.join(self.m_outDir, f"{name}_qq.png")
        pyplot.figure()
        stats.probplot(values, dist = "norm", plot = pyplot)
        pyplot.title(f"Q-Q Plot of {name}")
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()

    def _plot_cond_kde(self, name: str, values: numpy.ndarray, target: pandas.Series) -> None:
        path = os.path.join(self.m_outDir, f"{name}_cond_kde.png")
        pyplot.figure()
        maskTrue  = target.values
        maskFalse = ~target.values
        vTrue  = values[maskTrue]
        vFalse = values[maskFalse]
        if len(vTrue) > 1:
            kdeTrue = stats.gaussian_kde(vTrue)
            xTrue = numpy.linspace(vTrue.min(), vTrue.max(), 200)
            pyplot.plot(xTrue, kdeTrue(xTrue), label = "Churn=True", color = "red", linewidth = 1.5)
        if len(vFalse) > 1:
            kdeFalse = stats.gaussian_kde(vFalse)
            xFalse = numpy.linspace(vFalse.min(), vFalse.max(), 200)
            pyplot.plot(xFalse, kdeFalse(xFalse), label = "Churn=False", color = "blue", linewidth = 1.5)
        pyplot.title(f"Class-Conditional KDE of {name}")
        pyplot.xlabel(name)
        pyplot.ylabel("Density")
        pyplot.legend()
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()

    def _plot_scaling_compare(self, name: str, rawValues: numpy.ndarray) -> None:
        path = os.path.join(self.m_outDir, f"{name}_scaling_compare.png")
        fig, axes = pyplot.subplots(1, 3, figsize = (12, 4))
        for idx, (scalerName, scaler) in enumerate(SCALERS.items()):
            scaled = scaler.fit_transform(rawValues.reshape(-1, 1)).ravel()
            axes[idx].hist(scaled, bins = 30, density = True, alpha = 0.6, color = "steelblue", edgecolor = "white")
            kde = stats.gaussian_kde(scaled)
            x = numpy.linspace(scaled.min(), scaled.max(), 200)
            axes[idx].plot(x, kde(x), color = "darkred", linewidth = 1.5)
            axes[idx].set_title(scalerName)
            axes[idx].set_xlabel(name)
            axes[idx].set_ylabel("Density")
        fig.suptitle(f"Scaling Comparison for {name}")
        fig.tight_layout()
        fig.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close(fig)

    def __call__(self, features: pandas.DataFrame, target: pandas.Series, summary: pandas.DataFrame) -> None:
        os.makedirs(self.m_outDir, exist_ok = True)

        for col in features.columns:
            values = features[col].values.astype(float)
            self._plot_hist_kde(col, values)
            self._plot_qq(col, values)
            self._plot_cond_kde(col, values, target)

        topSkewed = summary.nlargest(4, columns = "skewness", keep = "all")
        for _, row in topSkewed.iterrows():
            col = row["feature"]
            rawValues = features[col].values.astype(float)
            self._plot_scaling_compare(col, rawValues)


if __name__ == "__main__":
    dataset = pandas.read_csv(IN_PATH)
    target = dataset["Churn"].copy()

    encoder = VecEncoder(maxDims = 3)
    dataset = encoder(dataset, "State")

    labelEnc = LabelEncoder(bindings = {"Yes": 1, "No": 0})
    dataset = labelEnc(dataset, "International plan")
    dataset = labelEnc(dataset, "Voice mail plan")

    # Store numeric features before filtering (for scaling comparison on unscaled data)
    corrFilter = CorrFilter(threshold = 0.95)
    dataset = corrFilter(dataset)

    # Separate target from features
    if "Churn" in dataset.columns:
        features = dataset.drop(columns = ["Churn"])
    else:
        features = dataset

    # Scale features
    scaler = StandardScaler()
    scaledArray = scaler.fit_transform(features)
    scaledDf = pandas.DataFrame(scaledArray, columns = features.columns, index = features.index)

    os.makedirs(OUT_DIR, exist_ok = True)

    analyzer = FeatureAnalyzer()
    summary = analyzer(scaledDf, target)

    plotter = DistributionPlotter(OUT_DIR)
    plotter(scaledDf, target, summary)

    summary.to_csv(SUMMARY_PATH, index = False)
    print(summary.to_string(index = False))
    print(f"\nSummary saved to {SUMMARY_PATH}")
    print(f"Plots saved to {OUT_DIR}/")