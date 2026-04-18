import os
import numpy
import pandas
from matplotlib import pyplot
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score

from Utils.Encoder import LabelEncoder
from Utils.Filter import CorrFilter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT_DIR, "Results", "Clustering")
IN_PATH  = os.path.join(ROOT_DIR, "Data", "churn-bigml-80.csv")
K_RANGE  = range(2, 11)

# Clustering Analyzers: KMeans, DBSCAN, GMM

class KMeansAnalyzer:
    def __init__(self, kRange = K_RANGE) -> None:
        self.m_kRange = kRange
        self.m_inertias = []
        self.m_silhouettes = []
        self.m_bestK = 2
        self.m_labels = None

    def __call__(self, data: numpy.ndarray) -> numpy.ndarray:
        self.m_inertias = []
        self.m_silhouettes = []
        for k in self.m_kRange:
            model = KMeans(n_clusters = k, random_state = 42, n_init = 10)
            labels = model.fit_predict(data)
            self.m_inertias.append(model.inertia_)
            self.m_silhouettes.append(silhouette_score(data, labels))

        self.m_bestK = list(self.m_kRange)[numpy.argmax(self.m_silhouettes)]
        print(f"KMeans best k = {self.m_bestK} (silhouette = {max(self.m_silhouettes):.4f})")

        model = KMeans(n_clusters = self.m_bestK, random_state = 42, n_init = 10)
        self.m_labels = model.fit_predict(data)
        return self.m_labels

    def plotElbow(self, path: str) -> None:
        pyplot.figure()
        pyplot.title("KMeans Elbow Method")
        pyplot.xlabel("k")
        pyplot.ylabel("Inertia")
        pyplot.plot(list(self.m_kRange), self.m_inertias, marker = "o")
        pyplot.savefig(path, dpi=600, bbox_inches="tight")
        pyplot.close()

    def plotSilhouette(self, path: str) -> None:
        pyplot.figure()
        pyplot.title("KMeans Silhouette Scores")
        pyplot.xlabel("k")
        pyplot.ylabel("Silhouette Score")
        pyplot.plot(list(self.m_kRange), self.m_silhouettes, marker = "o")
        pyplot.savefig(path, dpi=600, bbox_inches="tight")
        pyplot.close()


class DBSCANAnalyzer:
    EPS_GRID = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0]
    MIN_SAMPLES_GRID = [3, 5, 10, 15, 20]

    def __init__(self, epsGrid = None, minSamplesGrid = None) -> None:
        self.m_epsGrid = epsGrid if epsGrid is not None else self.EPS_GRID
        self.m_minSamplesGrid = minSamplesGrid if minSamplesGrid is not None else self.MIN_SAMPLES_GRID
        self.m_bestEps = None
        self.m_bestMinSamples = None
        self.m_labels = None
        self.m_distances = None
        self.m_gridResults = []

    def gridSearch(self, data: numpy.ndarray):
        bestSilhouette = -1.0
        bestEps = None
        bestMinSamples = None
        bestLabels = None

        for ms in self.m_minSamplesGrid:
            neighbors = NearestNeighbors(n_neighbors = ms)
            neighbors.fit(data)
            distances, _ = neighbors.kneighbors(data)
            kDists = numpy.sort(distances[:, -1])

            for eps in self.m_epsGrid:
                model = DBSCAN(eps = eps, min_samples = ms)
                labels = model.fit_predict(data)
                nClusters = len(set(labels)) - (1 if -1 in labels else 0)
                nNoise = int((labels == -1).sum())

                if nClusters < 2:
                    self.m_gridResults.append({
                        "eps": eps, "min_samples": ms,
                        "n_clusters": nClusters, "n_noise": nNoise,
                        "silhouette": float("nan")
                    })
                    continue

                mask = labels != -1
                sil = silhouette_score(data[mask], labels[mask])
                self.m_gridResults.append({
                    "eps": eps, "min_samples": ms,
                    "n_clusters": nClusters, "n_noise": nNoise,
                    "silhouette": sil
                })

                nTotal = len(data)
                noiseFraction = nNoise / nTotal
                if noiseFraction <= 0.5 and sil > bestSilhouette:
                    bestSilhouette = sil
                    bestEps = eps
                    bestMinSamples = ms
                    bestLabels = labels.copy()
                    self.m_distances = kDists

        # Fallback: if no config passed noise constraint, pick lowest-noise config with >= 2 clusters
        if bestEps is None:
            validResults = [r for r in self.m_gridResults if r["n_clusters"] >= 2]
            if validResults:
                best = min(validResults, key = lambda r: r["n_noise"])
                bestEps = best["eps"]
                bestMinSamples = best["min_samples"]
                # Re-fit to get labels
                model = DBSCAN(eps = bestEps, min_samples = bestMinSamples)
                bestLabels = model.fit_predict(data)
                # Recompute distances for plot
                neighbors = NearestNeighbors(n_neighbors = bestMinSamples)
                neighbors.fit(data)
                distances, _ = neighbors.kneighbors(data)
                self.m_distances = numpy.sort(distances[:, -1])

        return bestEps, bestMinSamples, bestLabels

    def __call__(self, data: numpy.ndarray) -> numpy.ndarray:
        self.m_bestEps, self.m_bestMinSamples, self.m_labels = self.gridSearch(data)

        nClusters = len(set(self.m_labels)) - (1 if -1 in self.m_labels else 0)
        nNoise = int((self.m_labels == -1).sum())

        nTotal = len(data)
        print("DBSCAN grid search complete.")
        print("Configs with >= 2 clusters:")
        for r in self.m_gridResults:
            if r["n_clusters"] >= 2:
                nf = r["n_noise"] / nTotal
                print(f"  eps={r['eps']}, min_samples={r['min_samples']}: "
                      f"clusters={r['n_clusters']}, noise={r['n_noise']} ({nf:.1%}), "
                      f"silhouette={r['silhouette']:.4f}")
        print(f"Selected: eps={self.m_bestEps}, min_samples={self.m_bestMinSamples} "
              f"-> {nClusters} clusters, {nNoise} noise points")
        return self.m_labels

    def plotKDistance(self, path: str) -> None:
        pyplot.figure()
        pyplot.title(f"DBSCAN K-Distance Plot (min_samples={self.m_bestMinSamples})")
        pyplot.xlabel("Points (sorted)")
        pyplot.ylabel(f"{self.m_bestMinSamples}-NN Distance")
        pyplot.plot(self.m_distances)
        pyplot.axhline(
            y = self.m_bestEps, color = "r", linestyle = "--",
            label = f"chosen eps={self.m_bestEps}"
        )
        pyplot.legend()
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()


class GMMAnalyzer:
    """
    Analyze clustering quality using Gaussian Mixture Models and Bayesian Information Criterion.
    """
    def __init__(self, kRange = K_RANGE) -> None:
        self.m_kRange = kRange
        self.m_bics = []
        self.m_bestK = 2
        self.m_labels = None

    def __call__(self, data: numpy.ndarray) -> numpy.ndarray:
        self.m_bics = []
        for k in self.m_kRange:
            model = GaussianMixture(n_components = k, random_state = 42)
            model.fit(data)
            self.m_bics.append(model.bic(data))

        self.m_bestK = list(self.m_kRange)[numpy.argmin(self.m_bics)]
        print(f"GMM best n_components = {self.m_bestK} (BIC = {min(self.m_bics):.2f})")

        model = GaussianMixture(n_components = self.m_bestK, random_state = 42)
        self.m_labels = model.fit_predict(data)
        return self.m_labels

    def plotBIC(self, path: str) -> None:
        pyplot.figure()
        pyplot.title("GMM BIC Scores")
        pyplot.xlabel("n_components")
        pyplot.ylabel("BIC")
        pyplot.plot(list(self.m_kRange), self.m_bics, marker = "o")
        pyplot.savefig(path, dpi=600, bbox_inches="tight")
        pyplot.close()


class ClusterProfiler:
    """
    Profile clusters by computing average feature values and churn rates per cluster.
    """
    def __init__(self, features: pandas.DataFrame, churn: pandas.Series) -> None:
        self.m_features = features
        self.m_churn = churn

    def profile(self, labels: numpy.ndarray, name: str) -> pandas.DataFrame:
        frame = self.m_features.copy()
        frame["Cluster"] = labels
        frame["Churn"] = self.m_churn.values
        # Filter noise points for DBSCAN
        frame = frame[frame["Cluster"] != -1]
        summary = frame.groupby("Cluster").mean()
        return summary

    def plotChurnRate(self, labels: numpy.ndarray, name: str, path: str,
                      highlightClusters: dict = None) -> None:
        frame = pandas.DataFrame({"Cluster": labels, "Churn": self.m_churn.values})
        frame = frame[frame["Cluster"] != -1]
        rates = frame.groupby("Cluster")["Churn"].mean().sort_values()
        sizes = frame.groupby("Cluster")["Churn"].count()[rates.index]

        cmap = pyplot.cm.get_cmap("tab10")
        # Skip tab10 index 3 (red) so highlight colour stays unique
        nonRedIndices = [0, 1, 2, 4, 5, 6, 7, 8, 9]
        colours = [cmap(nonRedIndices[i % len(nonRedIndices)]) for i in range(len(rates))]
        if highlightClusters:
            for i, clusterId in enumerate(rates.index):
                if clusterId in highlightClusters:
                    colours[i] = "#d32f2f"

        figWidth = max(6, len(rates) * 1.4)
        pyplot.figure(figsize = (figWidth, 5))
        pyplot.title(f"Churn Rate per Cluster ({name})")
        pyplot.xlabel("Cluster")
        pyplot.ylabel("Churn Rate")

        yTop = rates.max() * 1.25 if highlightClusters else rates.max() * 1.18
        pyplot.ylim(0, yTop)

        bars = pyplot.bar(x = [str(c) for c in rates.index], height = rates.values,
                          color = colours)

        # Annotate each bar: churn rate + sample count
        for bar, val, n in zip(bars, rates.values, sizes.values):
            pyplot.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + yTop * 0.02,
                f"{val:.1%}\nn={n}",
                ha = "center", va = "bottom", fontsize = 9, fontweight = "bold"
            )

        if highlightClusters:
            for bar, clusterId in zip(bars, rates.index):
                if clusterId in highlightClusters:
                    pyplot.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + yTop * 0.10,
                        highlightClusters[clusterId],
                        ha = "center", va = "bottom", fontsize = 8,
                        color = "#d32f2f", fontweight = "bold"
                    )

        pyplot.tight_layout()
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()

    def plotProfileHeatmap(self, labels: numpy.ndarray, name: str, path: str) -> None:
        frame = self.m_features.copy()
        frame["Cluster"] = labels
        frame = frame[frame["Cluster"] != -1]
        means = frame.groupby("Cluster")[self.m_features.columns.tolist()].mean()
        # z-score each column across clusters
        zscored = (means - means.mean()) / (means.std() + 1e-8)
        pyplot.figure(figsize = (14, max(4, len(means) * 0.8)))
        try:
            import seaborn
            seaborn.heatmap(zscored, annot = True, fmt = ".2f", cmap = "RdYlGn",
                            center = 0, linewidths = 0.5)
        except ImportError:
            im = pyplot.imshow(zscored.values, cmap = "RdYlGn", aspect = "auto")
            pyplot.colorbar(im)
            pyplot.xticks(range(len(zscored.columns)), zscored.columns, rotation = 45, ha = "right")
            pyplot.yticks(range(len(zscored)), [f"Cluster {i}" for i in zscored.index])
        pyplot.title(f"Cluster Profile Heatmap ({name}) -- z-scored feature means")
        pyplot.tight_layout()
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()

def computeMetrics(data: numpy.ndarray, labels: numpy.ndarray) -> dict:
    """
    Compare clustering quality using silhouette score, calinski-harabasz index, and davies-bouldin index.
    """
    mask = labels != -1
    if len(set(labels[mask])) < 2:
        return {"silhouette": float("nan"), "calinski": float("nan"), "davies": float("nan")}
    subset = data[mask]
    subLabels = labels[mask]
    return {
        "silhouette": silhouette_score(subset, subLabels),
        "calinski": calinski_harabasz_score(subset, subLabels),
        "davies": davies_bouldin_score(subset, subLabels),
    }


def plotMetricsComparison(allMetrics: dict, path: str) -> None:
    import math
    methods = list(allMetrics.keys())
    values = [allMetrics[m]["silhouette"] for m in methods]
    colours = ["#1976d2", "#d32f2f", "#388e3c"]  # blue, red, green

    # Wrap long labels with newlines for readability
    wrappedLabels = [m.replace(", ", ",\n") for m in methods]

    yMin = min(v for v in values if not math.isnan(v))
    yMax = max(v for v in values if not math.isnan(v))
    # Extend y-axis to show negative bars; add padding above and below
    padding = (yMax - yMin) * 0.3 if yMax != yMin else 0.05
    ylimBottom = min(yMin - padding, -0.05)
    ylimTop = yMax + padding

    pyplot.figure(figsize = (8, 5))
    bars = pyplot.bar(wrappedLabels, values, color = colours, width = 0.5)
    pyplot.axhline(y = 0, color = "black", linewidth = 0.8, linestyle = "-")
    pyplot.title("Clustering Quality: Silhouette Score Comparison")
    pyplot.ylabel("Silhouette Score")
    pyplot.ylim(ylimBottom, ylimTop)

    # Annotate each bar: positive bars → label above bar; negative bars → label below bar
    for bar, val in zip(bars, values):
        if math.isnan(val):
            continue
        xPos = bar.get_x() + bar.get_width() / 2
        if val >= 0:
            yPos = bar.get_height() + (ylimTop - ylimBottom) * 0.02
            va = "bottom"
        else:
            yPos = bar.get_height() - (ylimTop - ylimBottom) * 0.04
            va = "top"
        pyplot.text(xPos, yPos, f"{val:.3f}", ha = "center", va = va, fontsize = 10)

    pyplot.tight_layout()
    pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
    pyplot.close()


def plotPCAScatter(data: numpy.ndarray, labels: numpy.ndarray,
                   name: str, path: str) -> None:
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 2, random_state = 42)
    coords = pca.fit_transform(data)
    uniqueLabels = sorted(set(labels))
    pyplot.figure(figsize = (8, 6))
    for label in uniqueLabels:
        mask = labels == label
        labelName = "Noise" if label == -1 else f"Cluster {label}"
        colour = "lightgray" if label == -1 else None
        pyplot.scatter(coords[mask, 0], coords[mask, 1],
                       label = labelName, s = 10, alpha = 0.5, c = colour)
    pyplot.title(f"PCA 2D Scatter -- {name} clusters")
    pyplot.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    pyplot.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    pyplot.legend(markerscale = 2, fontsize = 8)
    pyplot.tight_layout()
    pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
    pyplot.close()


def plotFeatureScatter(dataset: pandas.DataFrame, labels: numpy.ndarray,
                       name: str, path: str) -> None:
    uniqueLabels = sorted(set(labels))
    pyplot.figure(figsize = (8, 6))
    for label in uniqueLabels:
        mask = labels == label
        labelName = "Noise" if label == -1 else f"Cluster {label}"
        colour = "lightgray" if label == -1 else None
        pyplot.scatter(
            dataset.loc[mask, "International plan"],
            dataset.loc[mask, "Total day charge"],
            label = labelName, s = 10, alpha = 0.5, c = colour
        )
    pyplot.title(f"International Plan vs Total Day Charge -- {name} clusters")
    pyplot.xlabel("International plan (encoded)")
    pyplot.ylabel("Total day charge")
    pyplot.legend(markerscale = 2, fontsize = 8)
    pyplot.tight_layout()
    pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
    pyplot.close()

if __name__ == "__main__":
    os.makedirs(OUT_PATH, exist_ok = True)

    # load and prepare data
    dataset = pandas.read_csv(IN_PATH)
    churn = dataset["Churn"].astype(int)
    dataset = dataset.drop(columns = ["Churn", "State", "Area code"])

    encoder = LabelEncoder({"Yes": 1, "No": 0})
    dataset = encoder(dataset, "International plan")
    dataset = encoder(dataset, "Voice mail plan")

    corrFilter = CorrFilter()
    dataset = corrFilter(dataset)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(dataset)

    print(f"Features after filtering: {dataset.shape[1]}")
    print(f"Columns: {list(dataset.columns)}")

    # KMeans
    kmeans = KMeansAnalyzer()
    kmLabels = kmeans(scaled)
    kmeans.plotElbow(f"{OUT_PATH}/KMeans_Elbow.png")
    kmeans.plotSilhouette(f"{OUT_PATH}/KMeans_Silhouette.png")

    # DBSCAN
    dbscan = DBSCANAnalyzer()
    dbLabels = dbscan(scaled)
    dbscan.plotKDistance(f"{OUT_PATH}/DBSCAN_KDistance.png")

    # GMM
    gmm = GMMAnalyzer()
    gmmLabels = gmm(scaled)
    gmm.plotBIC(f"{OUT_PATH}/GMM_BIC.png")

    # Profiling
    profiler = ClusterProfiler(dataset, churn)

    profiler.plotChurnRate(kmLabels, "KMeans", f"{OUT_PATH}/ClusterProfiles_KMeans.png")
    profiler.plotChurnRate(dbLabels, "DBSCAN", f"{OUT_PATH}/ClusterProfiles_DBSCAN.png",
                           highlightClusters = {4: "Silent\nChurners"})
    profiler.plotChurnRate(gmmLabels, "GMM", f"{OUT_PATH}/ClusterProfiles_GMM.png")

    # Cluster profile heatmaps
    profiler.plotProfileHeatmap(kmLabels, "KMeans", f"{OUT_PATH}/ClusterHeatmap_KMeans.png")
    profiler.plotProfileHeatmap(dbLabels, "DBSCAN", f"{OUT_PATH}/ClusterHeatmap_DBSCAN.png")
    profiler.plotProfileHeatmap(gmmLabels, "GMM", f"{OUT_PATH}/ClusterHeatmap_GMM.png")

    # PCA scatter by cluster
    plotPCAScatter(scaled, kmLabels, "KMeans", f"{OUT_PATH}/PCAScatter_KMeans.png")
    plotPCAScatter(scaled, dbLabels, "DBSCAN", f"{OUT_PATH}/PCAScatter_DBSCAN.png")
    plotPCAScatter(scaled, gmmLabels, "GMM", f"{OUT_PATH}/PCAScatter_GMM.png")

    # Feature scatter: International plan vs Total day charge
    plotFeatureScatter(dataset, kmLabels, "KMeans", f"{OUT_PATH}/FeatureScatter_KMeans.png")
    plotFeatureScatter(dataset, dbLabels, "DBSCAN", f"{OUT_PATH}/FeatureScatter_DBSCAN.png")
    plotFeatureScatter(dataset, gmmLabels, "GMM", f"{OUT_PATH}/FeatureScatter_GMM.png")

    # metrics comparison
    nDbClusters = len(set(dbLabels)) - (1 if -1 in dbLabels else 0)
    allMetrics = {
        f"K-Means (k={kmeans.m_bestK})": computeMetrics(scaled, kmLabels),
        f"DBSCAN (eps={dbscan.m_bestEps}, {nDbClusters} clusters)": computeMetrics(scaled, dbLabels),
        f"GMM ({gmm.m_bestK} components)": computeMetrics(scaled, gmmLabels),
    }
    plotMetricsComparison(allMetrics, f"{OUT_PATH}/Metrics_Comparison.png")

    # Summary
    lines = []
    lines.append("CLUSTERING ANALYSIS SUMMARY")
    import os; lines.append(f"Dataset: {os.path.basename(IN_PATH)}")
    lines.append(f"Features used: {dataset.shape[1]}")
    lines.append(f"Samples: {dataset.shape[0]}")
    lines.append("")

    lines.append(f"KMeans: best k = {kmeans.m_bestK}")
    kmProfile = profiler.profile(kmLabels, "KMeans")
    kmChurn = {k: round(float(v), 4) for k, v in kmProfile["Churn"].items()}
    lines.append(f"Churn rates per cluster: {kmChurn}")
    lines.append("")

    nClusters = len(set(dbLabels)) - (1 if -1 in dbLabels else 0)
    nNoise = (dbLabels == -1).sum()
    lines.append(f"DBSCAN: eps={dbscan.m_bestEps}, min_samples={dbscan.m_bestMinSamples}, "
                 f"clusters={nClusters}, noise={nNoise}")
    if nClusters >= 1:
        dbProfile = profiler.profile(dbLabels, "DBSCAN")
        dbChurn = {k: round(float(v), 4) for k, v in dbProfile["Churn"].items()}
        lines.append(f"Churn rates per cluster: {dbChurn}")
    lines.append("")

    lines.append(f"GMM: best n_components = {gmm.m_bestK}")
    gmmProfile = profiler.profile(gmmLabels, "GMM")
    gmmChurn = {k: round(float(v), 4) for k, v in gmmProfile["Churn"].items()}
    lines.append(f"Churn rates per cluster: {gmmChurn}")
    lines.append("")

    lines.append("Metrics Comparison:")
    for method, metrics in allMetrics.items():
        import math
        s = metrics['silhouette']
        c = metrics['calinski']
        d = metrics['davies']
        if math.isnan(s):
            lines.append(f"{method}: N/A (requires >= 2 clusters)")
        else:
            lines.append(f"{method}: silhouette={s:.4f}, "
                          f"calinski={c:.2f}, "
                          f"davies={d:.4f}")

    summary = "\n".join(lines)
    print(summary)
    with open(f"{OUT_PATH}/Summary.txt", "w") as f:
        f.write(summary)

    print(f"\nAll outputs saved to {OUT_PATH}/")
