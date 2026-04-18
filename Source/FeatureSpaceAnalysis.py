from __future__ import annotations
import os
import numpy
import pandas
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from Utils.Encoder import VecEncoder, LabelEncoder
from Utils.Filter import CorrFilter

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH     = os.path.join(ROOT_DIR, "Data", "churn-bigml-80.csv")
OUT_DIR     = os.path.join(ROOT_DIR, "Results", "FeatureSpace")


class PCAAnalyzer:
    def __init__(self) -> None:
        self.m_pca = None
        self.m_explained = None
        self.m_cumulative = None

    def __call__(self, data: numpy.ndarray) -> numpy.ndarray:
        self.m_pca = PCA()
        transformed = self.m_pca.fit_transform(data)
        self.m_explained = self.m_pca.explained_variance_ratio_
        self.m_cumulative = numpy.cumsum(self.m_explained)
        return transformed

    def plotScree(self, path: str) -> None:
        n = len(self.m_explained)
        x = numpy.arange(1, n + 1)
        pyplot.figure(figsize = (10, 5))
        pyplot.bar(x, self.m_explained, alpha = 0.7, label = "Individual")
        pyplot.plot(x, self.m_cumulative, marker = "o", color = "red", label = "Cumulative")
        pyplot.axhline(y = 0.90, color = "gray", linestyle = "--", label = "90% threshold")
        pyplot.axhline(y = 0.95, color = "black", linestyle = "--", label = "95% threshold")
        pyplot.xlabel("Principal Component")
        pyplot.ylabel("Explained Variance Ratio")
        pyplot.title("PCA Scree Plot")
        pyplot.xticks(x)
        pyplot.legend()
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()

    def plot2D(self, transformed: numpy.ndarray, target: pandas.Series, path: str) -> None:
        pyplot.figure()
        maskTrue = target.values
        maskFalse = ~target.values
        pyplot.scatter(transformed[maskFalse, 0], transformed[maskFalse, 1],
                       c = "blue", alpha = 0.4, label = "Churn=False", s = 10)
        pyplot.scatter(transformed[maskTrue, 0], transformed[maskTrue, 1],
                       c = "red", alpha = 0.4, label = "Churn=True", s = 10)
        pyplot.xlabel("PC1")
        pyplot.ylabel("PC2")
        pyplot.title("PCA: PC1 vs PC2")
        pyplot.legend()
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()


class FDRAnalyzer:
    def __init__(self) -> None:
        self.m_results = []
        self.m_df = None

    def __call__(self, features: pandas.DataFrame, target: pandas.Series) -> pandas.DataFrame:
        self.m_results = []
        maskTrue = target.values
        maskFalse = ~target.values
        for col in features.columns:
            vAll = features[col].values.astype(float)
            vTrue = vAll[maskTrue]
            vFalse = vAll[maskFalse]
            meanTrue = numpy.mean(vTrue)
            meanFalse = numpy.mean(vFalse)
            varTrue = numpy.var(vTrue, ddof = 1)
            varFalse = numpy.var(vFalse, ddof = 1)
            denom = varTrue + varFalse
            fdr = ((meanTrue - meanFalse) ** 2) / denom if denom > 0 else 0.0
            self.m_results.append({
                "feature":     col,
                "fdr":         fdr,
                "mean_churn":  meanTrue,
                "mean_nochurn": meanFalse,
                "var_churn":   varTrue,
                "var_nochurn": varFalse,
            })
        self.m_df = pandas.DataFrame(self.m_results).sort_values(
            by = "fdr", ascending = False
        ).reset_index(drop = True)
        return self.m_df

    def plotBar(self, path: str) -> None:
        pyplot.figure(figsize = (8, 6))
        pyplot.barh(self.m_df["feature"], self.m_df["fdr"], color = "steelblue")
        pyplot.xlabel("Fisher Discriminant Ratio")
        pyplot.ylabel("Feature")
        pyplot.title("FDR per Feature (sorted descending)")
        pyplot.gca().invert_yaxis()
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()


class TSNEAnalyzer:
    def __init__(self, perplexity: int = 30, randomState: int = 42) -> None:
        self.m_perplexity = perplexity
        self.m_randomState = randomState
        self.m_embedding = None

    def __call__(self, data: numpy.ndarray) -> numpy.ndarray:
        tsne = TSNE(n_components = 2, perplexity = self.m_perplexity,
                     random_state = self.m_randomState)
        self.m_embedding = tsne.fit_transform(data)
        return self.m_embedding

    def plot2D(self, target: pandas.Series, path: str) -> None:
        pyplot.figure()
        maskTrue = target.values
        maskFalse = ~target.values
        pyplot.scatter(self.m_embedding[maskFalse, 0], self.m_embedding[maskFalse, 1],
                       c = "blue", alpha = 0.4, label = "Churn=False", s = 10)
        pyplot.scatter(self.m_embedding[maskTrue, 0], self.m_embedding[maskTrue, 1],
                       c = "red", alpha = 0.4, label = "Churn=True", s = 10)
        pyplot.xlabel("t-SNE 1")
        pyplot.ylabel("t-SNE 2")
        pyplot.title("t-SNE 2D Projection")
        pyplot.legend()
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()


class LDAAnalyzer:
    def __init__(self) -> None:
        self.m_lda = None
        self.m_projection = None

    def __call__(self, data: numpy.ndarray, target: pandas.Series) -> numpy.ndarray:
        self.m_lda = LinearDiscriminantAnalysis(n_components = 1)
        self.m_projection = self.m_lda.fit_transform(data, target.values).ravel()
        return self.m_projection

    def plotProjection(self, target: pandas.Series, path: str) -> None:
        pyplot.figure()
        maskTrue = target.values
        maskFalse = ~target.values
        projTrue = self.m_projection[maskTrue]
        projFalse = self.m_projection[maskFalse]
        pyplot.hist(projFalse, bins = 30, alpha = 0.6, color = "blue", label = "Churn=False", density = True)
        pyplot.hist(projTrue, bins = 30, alpha = 0.6, color = "red", label = "Churn=True", density = True)
        pyplot.xlabel("LDA Projection")
        pyplot.ylabel("Density")
        pyplot.title("LDA 1D Projection by Churn")
        pyplot.legend()
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()


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

    # --- PCA ---
    pca = PCAAnalyzer()
    pcaTransformed = pca(scaledArray)
    pca.plotScree(os.path.join(OUT_DIR, "PCA_Scree.png"))
    pca.plot2D(pcaTransformed, target, os.path.join(OUT_DIR, "PCA_2D.png"))

    # --- FDR ---
    fdr = FDRAnalyzer()
    fdrDf = fdr(scaledDf, target)
    fdr.plotBar(os.path.join(OUT_DIR, "FDR_BarChart.png"))
    fdrDf.to_csv(os.path.join(OUT_DIR, "FDR_scores.csv"), index = False)

    # --- t-SNE ---
    tsne = TSNEAnalyzer(perplexity = 30, randomState = 42)
    tsneEmbedding = tsne(scaledArray)
    tsne.plot2D(target, os.path.join(OUT_DIR, "tSNE_2D.png"))

    # --- LDA ---
    lda = LDAAnalyzer()
    ldaProjection = lda(scaledArray, target)
    lda.plotProjection(target, os.path.join(OUT_DIR, "LDA_Projection.png"))

    # --- Summary ---
    print("Feature Space Analysis Complete")
    print(f"  PCA components: {len(pca.m_explained)}")
    n90 = int(numpy.searchsorted(pca.m_cumulative, 0.90)) + 1
    n95 = int(numpy.searchsorted(pca.m_cumulative, 0.95)) + 1
    print(f"  Components for 90% variance: {n90}")
    print(f"  Components for 95% variance: {n95}")
    print(f"  Top FDR features:")
    for _, row in fdrDf.head(5).iterrows():
        print(f"    {row['feature']}: {row['fdr']:.4f}")
    print(f"\nOutputs saved to {OUT_DIR}/")