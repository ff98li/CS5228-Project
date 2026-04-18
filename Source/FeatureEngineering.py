from __future__ import annotations
import os
import numpy
import pandas
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import KernelPCA
from Utils.Encoder import VecEncoder, LabelEncoder
from Utils.Filter import CorrFilter

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH     = os.path.join(ROOT_DIR, "Data", "churn-bigml-80.csv")
OUT_DIR     = os.path.join(ROOT_DIR, "Results", "FeatureEngineering")


class MIAnalyzer:
    def __init__(self) -> None:
        self.m_df = None

    def __call__(self, features: pandas.DataFrame, target: pandas.Series) -> pandas.DataFrame:
        scores = mutual_info_classif(features, target.values, random_state = 42)
        self.m_df = pandas.DataFrame({
            "feature":   features.columns,
            "mi_score":   scores,
        }).sort_values(by = "mi_score", ascending = False).reset_index(drop = True)
        return self.m_df

    def plotBar(self, path: str) -> None:
        pyplot.figure(figsize = (8, 6))
        pyplot.barh(self.m_df["feature"], self.m_df["mi_score"], color = "steelblue")
        pyplot.xlabel("Mutual Information Score")
        pyplot.ylabel("Feature")
        pyplot.title("Mutual Information per Feature")
        pyplot.gca().invert_yaxis()
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()


class InteractionAnalyzer:
    def __init__(self) -> None:
        self.m_df = None

    def __call__(self, features: pandas.DataFrame, target: pandas.Series, topN: int = 15) -> pandas.DataFrame:
        poly = PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)
        interactions = poly.fit_transform(features)
        interactionNames = poly.get_feature_names_out(features.columns)
        # Separate original features from pure interaction terms
        originalSet = set(features.columns)
        interactionDf = pandas.DataFrame(interactions, columns = interactionNames, index = features.index)
        # Keep only columns that are interaction terms (contain a space, i.e. "feat1 feat2")
        interactionCols = [c for c in interactionNames if c not in originalSet]
        interactionSubset = interactionDf[interactionCols]

        scores = mutual_info_classif(interactionSubset, target.values, random_state = 42)
        resultDf = pandas.DataFrame({
            "feature_pair": interactionCols,
            "mi_score":      scores,
        }).sort_values(by = "mi_score", ascending = False).reset_index(drop = True)

        self.m_df = resultDf.head(topN).reset_index(drop = True)
        return self.m_df

    def plotBar(self, path: str) -> None:
        pyplot.figure(figsize = (8, 6))
        pyplot.barh(self.m_df["feature_pair"], self.m_df["mi_score"], color = "coral")
        pyplot.xlabel("Mutual Information Score")
        pyplot.ylabel("Interaction")
        pyplot.title("Top 15 Interaction Terms by MI Score")
        pyplot.gca().invert_yaxis()
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()


class KernelPCAAnalyzer:
    def __init__(self, gamma: float = 0.1) -> None:
        self.m_gamma = gamma
        self.m_embedding = None

    def __call__(self, data: numpy.ndarray) -> numpy.ndarray:
        kpca = KernelPCA(n_components = 2, kernel = "rbf", gamma = self.m_gamma, random_state = 42)
        self.m_embedding = kpca.fit_transform(data)
        return self.m_embedding

    def plot2D(self, target: pandas.Series, path: str) -> None:
        pyplot.figure()
        maskTrue = target.values
        maskFalse = ~target.values
        pyplot.scatter(self.m_embedding[maskFalse, 0], self.m_embedding[maskFalse, 1],
                       c = "blue", alpha = 0.4, label = "Churn=False", s = 10)
        pyplot.scatter(self.m_embedding[maskTrue, 0], self.m_embedding[maskTrue, 1],
                       c = "red", alpha = 0.4, label = "Churn=True", s = 10)
        pyplot.xlabel("Kernel PC 1 (RBF)")
        pyplot.ylabel("Kernel PC 2 (RBF)")
        pyplot.title("Kernel PCA (RBF) 2D Projection")
        pyplot.legend()
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()


class SVDAnalyzer:
    def __init__(self) -> None:
        self.m_singularValues = None
        self.m_conditionNumber = None
        self.m_rank = None

    def __call__(self, data: numpy.ndarray) -> None:
        U, s, Vt = numpy.linalg.svd(data, full_matrices = False)
        self.m_singularValues = s
        self.m_conditionNumber = s[0] / s[-1]
        # Numerical rank: number of singular values above a small tolerance
        tol = s[0] * max(data.shape) * numpy.finfo(float).eps
        self.m_rank = int(numpy.sum(s > tol))

    def plotSingularValues(self, path: str) -> None:
        n = len(self.m_singularValues)
        x = numpy.arange(1, n + 1)
        pyplot.figure()
        pyplot.bar(x, self.m_singularValues, color = "steelblue")
        pyplot.xlabel("Component")
        pyplot.ylabel("Singular Value")
        pyplot.title("Singular Values of Feature Matrix")
        pyplot.xticks(x)
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()

    def saveSummary(self, path: str) -> None:
        lines = [
            "SVD SUMMARY",
            f"Condition number: {self.m_conditionNumber:.4f}",
            f"Numerical rank: {self.m_rank}",
            f"Top 5 singular values:",
        ]
        for i, sv in enumerate(self.m_singularValues[:5]):
            lines.append(f"  {i + 1}: {sv:.4f}")
        summary = "\n".join(lines)
        print(summary)
        with open(path, "w") as f:
            f.write(summary)


if __name__ == "__main__":
    # --- Load and preprocess (identical to DistribAnalysis.py) ---
    dataset = pandas.read_csv(IN_PATH)
    target = dataset["Churn"].copy()

    encoder = VecEncoder(maxDims = 3)
    dataset = encoder(dataset, "State")

    labelEnc = LabelEncoder(bindings = {"Yes": 1, "No": 0})
    dataset = labelEnc(dataset, "International plan")
    dataset = labelEnc(dataset, "Voice mail plan")

    corrFilter = CorrFilter(threshold = 0.95)
    dataset = corrFilter(dataset)

    if "Churn" in dataset.columns:
        features = dataset.drop(columns = ["Churn"])
    else:
        features = dataset

    scaler = StandardScaler()
    scaledArray = scaler.fit_transform(features)
    scaledDf = pandas.DataFrame(scaledArray, columns = features.columns, index = features.index)

    # --- Output directory ---
    os.makedirs(OUT_DIR, exist_ok = True)

    # --- Mutual Information ---
    mi = MIAnalyzer()
    miDf = mi(scaledDf, target)
    mi.plotBar(os.path.join(OUT_DIR, "MI_BarChart.png"))
    miDf.to_csv(os.path.join(OUT_DIR, "MI_scores.csv"), index = False)

    # --- Polynomial Interactions ---
    interactions = InteractionAnalyzer()
    interDf = interactions(scaledDf, target, topN = 15)
    interactions.plotBar(os.path.join(OUT_DIR, "Interactions_BarChart.png"))
    interDf.to_csv(os.path.join(OUT_DIR, "Interactions_Top15.csv"), index = False)

    # --- Kernel PCA (RBF) ---
    kpca = KernelPCAAnalyzer(gamma = 0.1)
    kpcaEmbedding = kpca(scaledArray)
    kpca.plot2D(target, os.path.join(OUT_DIR, "KernelPCA_RBF_2D.png"))

    # --- SVD Analysis ---
    svd = SVDAnalyzer()
    svd(scaledArray)
    svd.plotSingularValues(os.path.join(OUT_DIR, "SVD_SingularValues.png"))
    svd.saveSummary(os.path.join(OUT_DIR, "SVD_summary.txt"))

    print(f"\nAll outputs saved to {OUT_DIR}/")