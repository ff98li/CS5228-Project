from __future__ import annotations
import os
import numpy
import pandas
from matplotlib import pyplot
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)
from sklearn.base import clone
from Encoder import VecEncoder, LabelEncoder
from Filter import CorrFilter

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH  = os.path.join(ROOT_DIR, "Data", "churn-bigml-80.csv")
TEST_PATH   = os.path.join(ROOT_DIR, "Data", "churn-bigml-20.csv")
OUT_PATH    = os.path.join(ROOT_DIR, "Results", "GenAI")
SYNTH_PATH  = os.path.join(ROOT_DIR, "Data", "churn-gen-ai-test-data.csv")

# US state -> region mapping
STATE_REGIONS = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest", "IN": "Midwest", "IA": "Midwest", "KS": "Midwest",
    "MI": "Midwest", "MN": "Midwest", "MO": "Midwest", "NE": "Midwest",
    "ND": "Midwest", "OH": "Midwest", "SD": "Midwest", "WI": "Midwest",
    "AL": "South", "AR": "South", "DE": "South", "FL": "South", "GA": "South",
    "KY": "South", "LA": "South", "MD": "South", "MS": "South", "NC": "South",
    "OK": "South", "SC": "South", "TN": "South", "TX": "South", "VA": "South",
    "WV": "South", "DC": "South",
    "AZ": "West", "CA": "West", "CO": "West", "ID": "West", "MT": "West",
    "NV": "West", "NM": "West", "OR": "West", "UT": "West", "WA": "West",
    "WY": "West", "AK": "West", "HI": "West",
}

BN_FEATURES = [
    "State", "International plan", "Voice mail plan",
    "Number vmail messages", "Total day charge", "Total eve calls",
    "Total intl calls", "Total intl charge", "Customer service calls", "Churn",
]

BN_EDGES = [
    ("State", "Total day charge_bin"),
    ("State", "Customer service calls_bin"),
    ("International plan", "Total day charge_bin"),
    ("International plan", "Churn"),
    ("Voice mail plan", "Number vmail messages_bin"),
    ("Total day charge_bin", "Churn"),
    ("Customer service calls_bin", "Churn"),
    ("Total day charge_bin", "Total intl charge_bin"),
    ("Total intl calls_bin", "Total intl charge_bin"),
]

# Histogram comparison columns (raw CSV, 0-indexed D=3, E=4, G=6, H=7, I=8)
HIST_COLUMNS = [
    "International plan", "Voice mail plan",
    "Total day minutes", "Total day calls", "Total day charge",
]


class DataDiscretiser:
    def __init__(self) -> None:
        self.m_binEdges = {}

    def __call__(self, df: pandas.DataFrame) -> tuple:
        result = df[BN_FEATURES].copy()

        # Map State to region
        result["State"] = result["State"].map(STATE_REGIONS).fillna("Other")

        # Encode binary as string for pgmpy
        result["International plan"] = result["International plan"].map({1: "1", 0: "0", "Yes": "1", "No": "0"}).astype(str)
        result["Voice mail plan"] = result["Voice mail plan"].map({1: "1", 0: "0", "Yes": "1", "No": "0"}).astype(str)
        # Churn may be boolean; convert to "1"/"0"
        result["Churn"] = result["Churn"].map({True: "1", False: "0", 1: "1", 0: "0"}).fillna(result["Churn"]).astype(str)

        # Discretise Number vmail messages: 0, 1-3, 4+
        result["Number vmail messages_bin"] = pandas.cut(
            result["Number vmail messages"],
            bins = [-1, 0, 3, 100],
            labels = ["0", "1-3", "4+"]).astype(str)

        # Discretise Total day charge: 5 quantile bins
        result["Total day charge_bin"] = self._qcut(result["Total day charge"], "Total day charge", 5)

        # Discretise Total eve calls: 5 quantile bins
        result["Total eve calls_bin"] = self._qcut(result["Total eve calls"], "Total eve calls", 5)

        # Discretise Total intl calls: 3 bins
        result["Total intl calls_bin"] = self._qcut(result["Total intl calls"], "Total intl calls", 3)

        # Discretise Total intl charge: 5 quantile bins
        result["Total intl charge_bin"] = self._qcut(result["Total intl charge"], "Total intl charge", 5)

        # Discretise Customer service calls: 0-1, 2-3, 4+
        result["Customer service calls_bin"] = pandas.cut(
            result["Customer service calls"],
            bins = [-1, 1, 3, 100],
            labels = ["0-1", "2-3", "4+"]).astype(str)

        # Keep only the columns needed for BN
        bnCols = ["State", "International plan", "Voice mail plan",
                   "Number vmail messages_bin", "Total day charge_bin",
                   "Total intl calls_bin",
                   "Total intl charge_bin", "Customer service calls_bin", "Churn"]
        # Also store Total eve calls_bin for later post-processing
        self.m_eveCallsBin = result["Total eve calls_bin"].copy()
        return result[bnCols], self.m_binEdges

    def _qcut(self, series: pandas.Series, name: str, q: int) -> pandas.Series:
        binned, edges = pandas.qcut(series, q = q, retbins = True, duplicates = "drop")
        self.m_binEdges[name] = edges
        # Convert interval labels to string for pgmpy
        return binned.astype(str)

    def undiscretise(self, discrete_df: pandas.DataFrame) -> pandas.DataFrame:
        """Map bin labels back to continuous values by uniform sampling within intervals."""
        result = pandas.DataFrame()

        # State: keep as region string
        result["State"] = discrete_df["State"]

        # Binary features: convert back to int
        result["International plan"] = discrete_df["International plan"].astype(int)
        result["Voice mail plan"] = discrete_df["Voice mail plan"].astype(int)
        result["Churn"] = discrete_df["Churn"].astype(int)

        # Number vmail messages: map bins
        result["Number vmail messages"] = discrete_df["Number vmail messages_bin"].map(
            {"0": 0, "1-3": 2, "4+": 20}).astype(float)

        # Continuous features: uniform sample from bin intervals
        result["Total day charge"] = self._sample_from_bins(
            discrete_df["Total day charge_bin"], "Total day charge")
        # Total eve calls is not in the BN; sample from saved marginal bins
        if hasattr(self, "m_eveCallsBin") and self.m_eveCallsBin is not None:
            sampledBins = numpy.random.choice(self.m_eveCallsBin.values, size = len(discrete_df), replace = True)
            result["Total eve calls"] = self._sample_from_bins_values(
                sampledBins, "Total eve calls")
        else:
            result["Total eve calls"] = numpy.random.uniform(50, 200, size = len(discrete_df))
        result["Total intl calls"] = self._sample_from_bins(
            discrete_df["Total intl calls_bin"], "Total intl calls")
        result["Total intl charge"] = self._sample_from_bins(
            discrete_df["Total intl charge_bin"], "Total intl charge")
        result["Customer service calls"] = discrete_df["Customer service calls_bin"].map(
            {"0-1": 0, "2-3": 2, "4+": 5}).astype(float)

        return result

    def _sample_from_bins(self, binSeries: pandas.Series, name: str) -> numpy.ndarray:
        edges = self.m_binEdges.get(name, None)
        if edges is None:
            return numpy.zeros(len(binSeries))
        result = numpy.zeros(len(binSeries))
        for i, val in enumerate(binSeries):
            left, right = self._parse_interval(val, edges)
            result[i] = numpy.random.uniform(left, right)
        return result

    def _sample_from_bins_values(self, binValues: numpy.ndarray, name: str) -> numpy.ndarray:
        """Sample from bins given an array of interval label strings."""
        edges = self.m_binEdges.get(name, None)
        if edges is None:
            return numpy.zeros(len(binValues))
        result = numpy.zeros(len(binValues))
        for i, val in enumerate(binValues):
            left, right = self._parse_interval(val, edges)
            result[i] = numpy.random.uniform(left, right)
        return result

    def _parse_interval(self, label: str, edges: numpy.ndarray) -> tuple:
        """Parse a qcut interval label like '(0.12, 25.3]' into (left, right)."""
        try:
            # Remove brackets and split on comma
            clean = label.strip("[]()")
            parts = clean.split(",")
            left = float(parts[0].strip())
            right = float(parts[1].strip())
            return left, right
        except (ValueError, IndexError):
            # Fallback: return midpoint of edges
            mid = (edges[0] + edges[-1]) / 2
            return mid, mid


class BayesianSampler:
    def __init__(self) -> None:
        self.m_model = None
        self.m_discretiser = None
        self.m_useFallback = False

    def __call__(self, train_df: pandas.DataFrame, n_samples: int = 500) -> pandas.DataFrame:
        self.m_discretiser = DataDiscretiser()
        discrete_df, binEdges = self.m_discretiser(train_df)

        try:
            from pgmpy.models import DiscreteBayesianNetwork
            from pgmpy.estimators import MaximumLikelihoodEstimator
            from pgmpy.sampling import BayesianModelSampling

            print("Fitting Bayesian Network with pgmpy...")
            self.m_model = DiscreteBayesianNetwork(BN_EDGES)
            self.m_model.fit(discrete_df, estimator = MaximumLikelihoodEstimator)

            print(f"Sampling {n_samples} rows...")
            sampler = BayesianModelSampling(self.m_model)
            synthetic_discrete = sampler.forward_sample(size = n_samples)

            # Post-process: convert bin labels back to continuous
            synthetic_continuous = self.m_discretiser.undiscretise(synthetic_discrete)

        except Exception as e:
            print(f"pgmpy failed: {e}")
            print("Using GaussianCopula fallback.")
            self.m_useFallback = True
            synthetic_continuous = self._fallbackGaussianCopula(train_df, n_samples)

        return synthetic_continuous

    def _fallbackGaussianCopula(self, train_df: pandas.DataFrame, n_samples: int) -> pandas.DataFrame:
        try:
            from sdv.single_table import GaussianCopulaSynthesizer
            from sdv.metadata import SingleTableMetadata

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(train_df)
            synthesizer = GaussianCopulaSynthesizer(metadata)
            synthesizer.fit(train_df)
            synthetic = synthesizer.sample(num_rows = n_samples)
            return synthetic
        except ImportError:
            # Last resort: simple bootstrap sampling
            print("sdv not available. Using bootstrap sampling fallback.")
            indices = numpy.random.choice(len(train_df), size = n_samples, replace = True)
            return train_df.iloc[indices].reset_index(drop = True)


class HistogramComparator:
    def __init__(self) -> None:
        pass

    def __call__(self, real_df: pandas.DataFrame, synthetic_df: pandas.DataFrame) -> None:
        fig, axes = pyplot.subplots(2, 3, figsize = (15, 10))
        axes = axes.ravel()

        for idx, col in enumerate(HIST_COLUMNS):
            if idx >= 6:
                break
            ax = axes[idx]
            if col in real_df.columns and col in synthetic_df.columns:
                realVals = real_df[col].dropna()
                synthVals = synthetic_df[col].dropna()
                if col in ("International plan", "Voice mail plan"):
                    # Binary: side-by-side bar
                    realCounts = realVals.value_counts().sort_index()
                    synthCounts = synthVals.value_counts().sort_index()
                    x = numpy.arange(len(realCounts))
                    width = 0.35
                    ax.bar(x - width / 2, realCounts.values, width, label = "Real test", color = "blue", alpha = 0.7)
                    ax.bar(x + width / 2, synthCounts.reindex(realCounts.index, fill_value = 0).values,
                           width, label = "Synthetic", color = "orange", alpha = 0.7)
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(v) for v in realCounts.index])
                else:
                    # Continuous: overlapping KDE
                    kdeReal = stats.gaussian_kde(realVals.values.astype(float))
                    kdeSynth = stats.gaussian_kde(synthVals.values.astype(float))
                    xmin = min(realVals.min(), synthVals.min())
                    xmax = max(realVals.max(), synthVals.max())
                    x = numpy.linspace(xmin, xmax, 200)
                    ax.plot(x, kdeReal(x), label = "Real test", color = "blue")
                    ax.plot(x, kdeSynth(x), label = "Synthetic", color = "orange")
                ax.set_title(col)
                ax.legend(fontsize = 7)

        # Hide unused subplot
        if len(HIST_COLUMNS) < 6:
            axes[5].set_visible(False)

        fig.suptitle("Real vs Synthetic Data Distribution")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_PATH, "histogram_comparison.png"), dpi = 600, bbox_inches = "tight")
        pyplot.close(fig)


class ModelEvaluator:
    def __init__(self) -> None:
        self.m_model = None

    def _loadAndPreprocess(self, path: str):
        dataset = pandas.read_csv(path)
        target = dataset["Churn"].astype(int)
        dataset = dataset.drop(columns = ["Churn"])

        encoder = VecEncoder(maxDims = 3)
        dataset = encoder(dataset, "State")

        labelEnc = LabelEncoder(bindings = {"Yes": 1, "No": 0})
        dataset = labelEnc(dataset, "International plan")
        dataset = labelEnc(dataset, "Voice mail plan")

        corrFilter = CorrFilter(threshold = 0.95)
        dataset = corrFilter(dataset)

        # Drop zero-MI features
        for col in ["Account length", "Area code", "Total day calls",
                    "Total eve charge", "Total night calls", "Total night charge"]:
            if col in dataset.columns:
                dataset = dataset.drop(columns = [col])

        keep = [c for c in ["International plan", "Number vmail messages", "Total day charge",
                            "Total eve calls", "Total intl calls", "Total intl charge",
                            "Customer service calls", "State|PC0", "State|PC1", "State|PC2"]
                if c in dataset.columns]
        return dataset[keep], target

    def _trainBestModel(self, X_train, y_train):
        """Train RF condition D (best model from step_2)."""
        base = RandomForestClassifier(n_estimators = 100, random_state = 42)
        # Hyperparams from step_2: max_depth=None, min_samples_leaf=5
        base.set_params(max_depth = None, min_samples_leaf = 5)
        # Condition D: CalibratedClassifierCV + 3-seed ensemble
        ensemble = []
        for seed in [0, 1, 2]:
            m = clone(base)
            m.set_params(random_state = seed)
            cal = CalibratedClassifierCV(estimator = m, method = "sigmoid", cv = 3)
            cal.fit(X_train, y_train)
            ensemble.append(cal)
        self.m_model = _EnsembleWrapper(ensemble)

    def __call__(self, synthetic_df: pandas.DataFrame) -> dict:
        # Train on real training data
        X_train, y_train = self._loadAndPreprocess(TRAIN_PATH)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self._trainBestModel(X_train_scaled, y_train)

        # Evaluate on real test set
        X_test_real, y_test_real = self._loadAndPreprocess(TEST_PATH)
        X_test_real_scaled = scaler.transform(X_test_real)
        real_metrics = self._evaluate(X_test_real_scaled, y_test_real)

        # Evaluate on synthetic data
        # Synthetic CSV uses raw column names; need to preprocess similarly
        X_synth, y_synth = self._preprocessSynthetic(synthetic_df)
        X_synth_scaled = scaler.transform(X_synth)
        synth_metrics = self._evaluate(X_synth_scaled, y_synth)

        # Save comparison
        comparison = pandas.DataFrame([
            {"dataset": "Real test", **real_metrics},
            {"dataset": "Synthetic", **synth_metrics},
        ])
        comparison.to_csv(os.path.join(OUT_PATH, "model_comparison.csv"), index = False)

        print("\nModel Comparison (RF/D):")
        print(comparison.to_string(index = False))

        return {"real_test_f1": real_metrics["f1"], "synthetic_f1": synth_metrics["f1"]}

    def _evaluate(self, X, y) -> dict:
        y_pred = self.m_model.predict(X)
        y_proba = self.m_model.predict_proba(X)[:, 1]
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, pos_label = 1, zero_division = 0),
            "recall": recall_score(y, y_pred, pos_label = 1, zero_division = 0),
            "f1": f1_score(y, y_pred, pos_label = 1, zero_division = 0),
            "auc_roc": roc_auc_score(y, y_proba),
        }

    def _preprocessSynthetic(self, synth_df: pandas.DataFrame) -> tuple:
        """Preprocess synthetic data the same way as training data."""
        target = synth_df["Churn"].astype(int)
        dataset = synth_df.drop(columns = ["Churn"])

        encoder = VecEncoder(maxDims = 3)
        dataset = encoder(dataset, "State")

        labelEnc = LabelEncoder(bindings = {"Yes": 1, "No": 0})
        dataset = labelEnc(dataset, "International plan")
        dataset = labelEnc(dataset, "Voice mail plan")

        corrFilter = CorrFilter(threshold = 0.95)
        dataset = corrFilter(dataset)

        for col in ["Account length", "Area code", "Total day calls",
                    "Total eve charge", "Total night calls", "Total night charge"]:
            if col in dataset.columns:
                dataset = dataset.drop(columns = [col])

        keep = [c for c in ["International plan", "Number vmail messages", "Total day charge",
                            "Total eve calls", "Total intl calls", "Total intl charge",
                            "Customer service calls", "State|PC0", "State|PC1", "State|PC2"]
                if c in dataset.columns]
        return dataset[keep], target


class _EnsembleWrapper:
    def __init__(self, models: list) -> None:
        self.m_models = models
        self.classes_ = models[0].classes_

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        probas = [m.predict_proba(X) for m in self.m_models]
        return numpy.mean(probas, axis = 0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


if __name__ == "__main__":
    os.makedirs(OUT_PATH, exist_ok = True)

    # --- Load training data (raw) ---
    train_raw = pandas.read_csv(TRAIN_PATH)
    test_raw = pandas.read_csv(TEST_PATH)

    # --- Bayesian Network Sampling ---
    sampler = BayesianSampler()
    synthetic = sampler(train_raw, n_samples = 500)

    print(f"\nSynthetic data shape: {synthetic.shape}")
    print(f"Synthetic churn rate: {synthetic['Churn'].mean():.4f}")
    real_churn_rate = train_raw["Churn"].mean()
    if abs(synthetic["Churn"].mean() - real_churn_rate) > 0.05:
        print(f"WARNING: Synthetic churn rate deviates >5pp from real ({real_churn_rate:.4f})")

    # --- Save synthetic CSV ---
    # Reconstruct full CSV with all raw columns (fill missing ones with defaults)
    output = synthetic.copy()
    # Add columns from raw CSV that aren't in synthetic
    for col in train_raw.columns:
        if col not in output.columns:
            if col == "Area code":
                output[col] = 415  # default area code
            elif col == "Account length":
                output[col] = numpy.random.randint(20, 250, size = len(output))
            elif col == "Total day minutes":
                # Derive from Total day charge (charge ≈ minutes * 0.17)
                output[col] = output["Total day charge"] / 0.17
            elif col == "Total day calls":
                output[col] = numpy.random.randint(50, 170, size = len(output))
            elif col == "Total eve minutes":
                output[col] = numpy.random.randint(50, 370, size = len(output))
            elif col == "Total eve charge":
                output[col] = output.get("Total eve charge",
                    numpy.random.uniform(10, 30, size = len(output)))
            elif col == "Total night minutes":
                output[col] = numpy.random.randint(50, 400, size = len(output))
            elif col == "Total night calls":
                output[col] = numpy.random.randint(50, 180, size = len(output))
            elif col == "Total night charge":
                output[col] = numpy.random.uniform(5, 18, size = len(output))
            elif col == "Total intl minutes":
                output[col] = output.get("Total intl charge",
                    numpy.random.uniform(0, 5, size = len(output))) / 0.27
            elif col == "State":
                pass  # already present
            elif col == "Churn":
                output[col] = output["Churn"].astype(bool)
            else:
                output[col] = 0

    # Reorder columns to match raw CSV
    colOrder = [c for c in train_raw.columns if c in output.columns]
    output = output[colOrder]

    # Map binary back to Yes/No for consistency with raw CSV
    if "International plan" in output.columns:
        output["International plan"] = output["International plan"].map({1: "Yes", 0: "No"}).fillna(output["International plan"])
    if "Voice mail plan" in output.columns:
        output["Voice mail plan"] = output["Voice mail plan"].map({1: "Yes", 0: "No"}).fillna(output["Voice mail plan"])

    output.to_csv(SYNTH_PATH, index = False)
    print(f"Synthetic data saved to {SYNTH_PATH}")

    # --- Histogram Comparison ---
    comparator = HistogramComparator()
    comparator(test_raw, output)
    print(f"Histogram comparison saved to {OUT_PATH}/histogram_comparison.png")

    # --- Model Evaluation ---
    evaluator = ModelEvaluator()
    results = evaluator(output)
    print(f"\nModel comparison saved to {OUT_PATH}/model_comparison.csv")
    print(f"All outputs saved to {OUT_PATH}/")