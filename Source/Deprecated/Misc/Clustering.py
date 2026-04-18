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

from Encoder import LabelEncoder
from Filter import CorrFilter

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
    """
    Analyze clustering quality using DBSCAN
    """
    def __init__(self, minSamples: int = 5) -> None:
        self.m_minSamples = minSamples
        self.m_eps = None
        self.m_labels = None
        self.m_distances = None

    def estimateEps(self, data: numpy.ndarray) -> float:
        neighbors = NearestNeighbors(n_neighbors = self.m_minSamples)
        neighbors.fit(data)
        distances, _ = neighbors.kneighbors(data)
        self.m_distances = numpy.sort(distances[:, -1])
        # Use knee heuristic: point of maximum curvature
        diffs = numpy.diff(self.m_distances)
        eps = self.m_distances[numpy.argmax(diffs)]
        return eps

    def __call__(self, data: numpy.ndarray) -> numpy.ndarray:
        self.m_eps = self.estimateEps(data)
        print(f"DBSCAN estimated eps = {self.m_eps:.4f}")

        model = DBSCAN(eps = self.m_eps, min_samples = self.m_minSamples)
        self.m_labels = model.fit_predict(data)

        nClusters = len(set(self.m_labels)) - (1 if -1 in self.m_labels else 0)
        nNoise = (self.m_labels == -1).sum()
        print(f"DBSCAN found {nClusters} clusters, {nNoise} noise points")
        return self.m_labels

    def plotKDistance(self, path: str) -> None:
        pyplot.figure()
        pyplot.title("DBSCAN K-Distance Plot")
        pyplot.xlabel("Points (sorted)")
        pyplot.ylabel(f"{self.m_minSamples}-NN Distance")
        pyplot.plot(self.m_distances)
        pyplot.axhline(y = self.m_eps, color = "r", linestyle = "--", label = f"eps = {self.m_eps:.4f}")
        pyplot.legend()
        pyplot.savefig(path, dpi=600, bbox_inches="tight")
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

    def plotChurnRate(self, labels: numpy.ndarray, name: str, path: str) -> None:
        frame = pandas.DataFrame({"Cluster": labels, "Churn": self.m_churn.values})
        frame = frame[frame["Cluster"] != -1]
        rates = frame.groupby("Cluster")["Churn"].mean()
        pyplot.figure()
        pyplot.title(f"Churn Rate per Cluster ({name})")
        pyplot.xlabel("Cluster")
        pyplot.ylabel("Churn Rate")
        pyplot.bar(x = [str(c) for c in rates.index], height = rates.values)
        pyplot.savefig(path, dpi=600, bbox_inches="tight")
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
    methods = list(allMetrics.keys())
    metricNames = ["silhouette", "calinski", "davies"]
    fig, axes = pyplot.subplots(1, 3, figsize = (15, 5))
    for idx, metric in enumerate(metricNames):
        values = [allMetrics[m][metric] for m in methods]
        axes[idx].bar(methods, values)
        axes[idx].set_title(metric.capitalize())
        axes[idx].set_ylabel(metric)
    pyplot.tight_layout()
    pyplot.savefig(path, dpi=600, bbox_inches="tight")
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
    profiler.plotChurnRate(dbLabels, "DBSCAN", f"{OUT_PATH}/ClusterProfiles_DBSCAN.png")
    profiler.plotChurnRate(gmmLabels, "GMM", f"{OUT_PATH}/ClusterProfiles_GMM.png")

    # metrics comparison
    allMetrics = {
        "KMeans": computeMetrics(scaled, kmLabels),
        "DBSCAN": computeMetrics(scaled, dbLabels),
        "GMM": computeMetrics(scaled, gmmLabels),
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
    lines.append(f"DBSCAN: eps = {dbscan.m_eps:.4f}, clusters = {nClusters}, noise = {nNoise}")
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
