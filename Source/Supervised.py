from __future__ import annotations
import os
import numpy
import pandas
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              RocCurveDisplay)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from Encoder import VecEncoder, LabelEncoder
from Filter import CorrFilter

ROOT_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH     = os.path.join(ROOT_DIR, "Data", "churn-bigml-80.csv")
TEST_PATH      = os.path.join(ROOT_DIR, "Data", "churn-bigml-20.csv")
OUT_PATH       = os.path.join(ROOT_DIR, "Results", "Supervised")

FEATURES_10 = [
    "International plan", "Number vmail messages", "Total day charge",
    "Total eve calls", "Total intl calls", "Total intl charge",
    "Customer service calls", "State|PC0", "State|PC1", "State|PC2",
]

DROP_FEATURES = [
    "Account length", "Area code", "Total day calls",
    "Total eve charge", "Total night calls", "Total night charge",
]

SKEWED_FEATURES = [
    "International plan", "Number vmail messages",
    "Total intl calls", "Customer service calls",
]

INTERACTION_PAIRS = [
    ("International plan", "Total day charge", "Intl_x_DayCharge"),
    ("Total day charge", "Customer service calls", "DayCharge_x_SvcCalls"),
    ("International plan", "Customer service calls", "Intl_x_SvcCalls"),
]

CONDITIONS = ["A", "B", "C", "D"]
MODELS     = ["RF", "XGBoost", "LR"]


class DataLoader:
    def __init__(self) -> None:
        self.m_vecEncoder = None
        self.m_labelEnc = None
        self.m_corrFilter = None
        self.m_trainColumns = None

    def _encodeTrain(self, dataset: pandas.DataFrame) -> pandas.DataFrame:
        # Fit VecEncoder on train
        self.m_vecEncoder = VecEncoder(maxDims = 3)
        dataset = self.m_vecEncoder(dataset, "State")

        # Fit LabelEncoder on train
        self.m_labelEnc = LabelEncoder(bindings = {"Yes": 1, "No": 0})
        dataset = self.m_labelEnc(dataset, "International plan")
        dataset = self.m_labelEnc(dataset, "Voice mail plan")

        # Fit CorrFilter on train
        self.m_corrFilter = CorrFilter(threshold = 0.95)
        dataset = self.m_corrFilter(dataset)
        self.m_trainColumns = list(dataset.columns)
        return dataset

    def _encodeTest(self, dataset: pandas.DataFrame) -> pandas.DataFrame:
        # Apply already-fit VecEncoder to test
        dataset = self.m_vecEncoder(dataset, "State")

        # Apply already-fit LabelEncoder to test
        dataset = self.m_labelEnc(dataset, "International plan")
        dataset = self.m_labelEnc(dataset, "Voice mail plan")

        # Apply CorrFilter using same dropped columns as train
        for col in self.m_trainColumns:
            if col not in dataset.columns:
                # Column was already dropped by encoding, skip
                pass
        # Drop columns that CorrFilter removed during training
        # Re-run the filter logic: remove any column that is NOT in trainColumns (except Churn)
        keepCols = [c for c in dataset.columns if c in self.m_trainColumns or c == "Churn"]
        dataset = dataset[keepCols]

        return dataset

    def _dropZeroMI(self, dataset: pandas.DataFrame) -> pandas.DataFrame:
        for col in DROP_FEATURES:
            if col in dataset.columns:
                dataset = dataset.drop(columns = [col])
        return dataset

    def __call__(self) -> tuple:
        # Load and fit on train
        trainData = pandas.read_csv(TRAIN_PATH)
        y_train = trainData["Churn"].astype(int)
        trainData = trainData.drop(columns = ["Churn"])
        trainData = self._encodeTrain(trainData)

        # Load and transform test
        testData = pandas.read_csv(TEST_PATH)
        y_test = testData["Churn"].astype(int)
        testData = testData.drop(columns = ["Churn"])
        testData = self._encodeTest(testData)

        # Drop zero-MI features
        X_train = self._dropZeroMI(trainData)
        X_test  = self._dropZeroMI(testData)

        # Ensure same column order
        cols = [c for c in FEATURES_10 if c in X_train.columns]
        X_train = X_train[cols]
        X_test  = X_test[cols]
        return X_train, y_train, X_test, y_test


class FeatureSelector:
    def __init__(self) -> None:
        pass

    def __call__(self, X: pandas.DataFrame, isLR: bool = False) -> pandas.DataFrame:
        result = X[FEATURES_10].copy()
        if isLR:
            for f1, f2, name in INTERACTION_PAIRS:
                result[name] = result[f1].values.astype(float) * result[f2].values.astype(float)
        return result


class Preprocessor:
    def __init__(self) -> None:
        self.m_transformer = None

    def _buildTransformer(self, featureNames: list) -> ColumnTransformer:
        skewed = [f for f in SKEWED_FEATURES if f in featureNames]
        standard = [f for f in featureNames if f not in skewed]
        # Interaction columns go to standard scaler
        interactionCols = [name for _, _, name in INTERACTION_PAIRS]
        standard = [f for f in standard if f not in interactionCols]
        standard = standard + [n for n in interactionCols if n in featureNames]

        transformer = ColumnTransformer(
            transformers = [
                ("robust", RobustScaler(), skewed),
                ("standard", StandardScaler(), standard),
            ],
            remainder = "drop",
        )
        return transformer

    def fitTransform(self, X: pandas.DataFrame, isLR: bool = False) -> numpy.ndarray:
        if isLR:
            self.m_transformer = self._buildTransformer(list(X.columns))
            return self.m_transformer.fit_transform(X)
        # RF and XGBoost: no scaling needed, return raw values
        return X.values.astype(float)

    def transform(self, X: pandas.DataFrame, isLR: bool = False) -> numpy.ndarray:
        if isLR:
            return self.m_transformer.transform(X)
        return X.values.astype(float)


class ModelFactory:
    def __init__(self) -> None:
        pass

    def __call__(self, name: str, seed: int = 42):
        if name == "RF":
            return RandomForestClassifier(n_estimators = 100, random_state = seed)
        elif name == "XGBoost":
            return XGBClassifier(
                n_estimators = 200, early_stopping_rounds = 20,
                random_state = seed, eval_metric = "logloss", verbosity = 0,
            )
        elif name == "LR":
            return LogisticRegression(max_iter = 1000, random_state = seed)
        else:
            raise ValueError(f"Unknown model: {name}")


class AblationConditions:
    def __init__(self) -> None:
        pass

    def __call__(self, baseModel, condition: str, modelName: str,
                 X_train, y_train, X_test_raw, isLR: bool = False):
        if condition == "A":
            model = self._clone(baseModel)
            if modelName == "XGBoost":
                model.set_params(early_stopping_rounds = 20)
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size = 0.2, random_state = 42, stratify = y_train)
                model.fit(X_tr, y_tr, eval_set = [(X_val, y_val)])
            else:
                model.fit(X_train, y_train)
            return model

        elif condition == "B":
            if modelName == "XGBoost":
                # scale_pos_weight must be in constructor, not set_params
                model = XGBClassifier(
                    n_estimators = 200, early_stopping_rounds = 20,
                    random_state = 42, eval_metric = "logloss", verbosity = 0,
                    scale_pos_weight = 5.0,
                )
            else:
                model = self._clone(baseModel)
                if modelName == "RF":
                    model.set_params(class_weight = "balanced")
                elif modelName == "LR":
                    model.set_params(class_weight = "balanced")
            if modelName == "XGBoost":
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_train, y_train, test_size = 0.2, random_state = 42, stratify = y_train)
                model.fit(X_tr, y_tr, eval_set = [(X_val, y_val)])
            else:
                model.fit(X_train, y_train)
            return model

        elif condition == "C":
            smote = SMOTE(random_state = 42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            model = self._clone(baseModel)
            if modelName == "XGBoost":
                model.set_params(early_stopping_rounds = 20)
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_res, y_res, test_size = 0.2, random_state = 42, stratify = y_res)
                model.fit(X_tr, y_tr, eval_set = [(X_val, y_val)])
            else:
                model.fit(X_res, y_res)
            return model

        elif condition == "D":
            # Condition D: Calibration + multi-seed ensemble for RF/XGBoost
            if modelName in ("RF", "XGBoost"):
                return self._fitConditionDEnsemble(baseModel, modelName, X_train, y_train)
            else:
                cal = CalibratedClassifierCV(
                    estimator = self._clone(baseModel), method = "sigmoid", cv = 3)
                cal.fit(X_train, y_train)
                return cal

        else:
            raise ValueError(f"Unknown condition: {condition}")

    def _clone(self, model):
        from sklearn.base import clone
        cloned = clone(model)
        # Disable early stopping on cloned model — it needs eval_set which is not
        # available inside CalibratedClassifierCV or GridSearchCV
        if hasattr(cloned, "early_stopping_rounds"):
            cloned.set_params(early_stopping_rounds = None)
        return cloned

    def _fitConditionDEnsemble(self, baseModel, modelName, X_train, y_train):
        ensemble = []
        for seed in [0, 1, 2]:
            model = self._clone(baseModel)
            model.set_params(random_state = seed)
            cal = CalibratedClassifierCV(
                estimator = model, method = "sigmoid", cv = 3)
            cal.fit(X_train, y_train)
            ensemble.append(cal)
        return _EnsembleWrapper(ensemble)


class _EnsembleWrapper:
    """Wraps multiple fitted models; averages predict_proba, thresholds at 0.5."""
    def __init__(self, models: list) -> None:
        self.m_models = models
        self.classes_ = models[0].classes_

    def fit(self, X, y):
        # Already fitted; no-op for sklearn compatibility
        return self

    def predict_proba(self, X):
        probas = [m.predict_proba(X) for m in self.m_models]
        avg = numpy.mean(probas, axis = 0)
        return avg

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


class Evaluator:
    def __init__(self) -> None:
        self.m_results = []

    def __call__(self, model, X_test, y_test, modelName: str, condition: str) -> dict:
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label = 1, zero_division = 0)
        rec  = recall_score(y_test, y_pred, pos_label = 1, zero_division = 0)
        f1   = f1_score(y_test, y_pred, pos_label = 1, zero_division = 0)
        auc  = roc_auc_score(y_test, y_proba)

        result = {
            "model": modelName, "condition": condition,
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc_roc": auc,
        }
        self.m_results.append(result)

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        self._plotConfusionMatrix(cm, modelName, condition)

        return result

    def _plotConfusionMatrix(self, cm, modelName, condition):
        path = os.path.join(OUT_PATH, f"confusion_matrix_{modelName}_{condition}.png")
        pyplot.figure()
        pyplot.imshow(cm, interpolation = "nearest", cmap = pyplot.cm.Blues)
        pyplot.title(f"Confusion Matrix: {modelName} / Condition {condition}")
        pyplot.colorbar()
        tickMarks = [0, 1]
        pyplot.xticks(tickMarks, ["No Churn", "Churn"])
        pyplot.yticks(tickMarks, ["No Churn", "Churn"])
        for i in range(2):
            for j in range(2):
                pyplot.text(j, i, format(cm[i, j], "d"),
                            horizontalalignment = "center",
                            color = "white" if cm[i, j] > cm.max() / 2 else "black")
        pyplot.ylabel("True label")
        pyplot.xlabel("Predicted label")
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()

    def plotRocCurves(self, allResults, rocData):
        path = os.path.join(OUT_PATH, "roc_curves.png")
        pyplot.figure(figsize = (8, 6))
        conditionColors = {"A": "blue", "B": "green", "C": "orange", "D": "red"}
        modelStyles = {"RF": "-", "XGBoost": "--", "LR": "-."}
        for (modelName, condition), (fpr, tpr) in rocData.items():
            color = conditionColors.get(condition, "black")
            style = modelStyles.get(modelName, "-")
            label = f"{modelName}/{condition}"
            pyplot.plot(fpr, tpr, linestyle = style, color = color, label = label, alpha = 0.8)
        pyplot.plot([0, 1], [0, 1], "k--", alpha = 0.3)
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("True Positive Rate")
        pyplot.title("ROC Curves")
        pyplot.legend(fontsize = 7, loc = "lower right")
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()

    def saveResultsCsv(self):
        df = pandas.DataFrame(self.m_results)
        df.to_csv(os.path.join(OUT_PATH, "results_table.csv"), index = False)
        return df


class HyperparamTuner:
    def __init__(self) -> None:
        pass

    def __call__(self, modelName, baseModel, X, y):
        if modelName == "RF":
            paramGrid = {"max_depth": [None, 5, 10, 20], "min_samples_leaf": [1, 5, 10]}
        elif modelName == "XGBoost":
            paramGrid = {"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.3]}
            # Disable early stopping during CV (no eval_set available in CV splits)
            baseModel.set_params(early_stopping_rounds = None)
        elif modelName == "LR":
            paramGrid = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}
        else:
            raise ValueError(f"Unknown model: {modelName}")

        grid = GridSearchCV(baseModel, paramGrid, cv = 5, scoring = "f1", n_jobs = -1)
        grid.fit(X, y)
        best = grid.best_estimator_
        # Re-enable early stopping for XGBoost
        if modelName == "XGBoost":
            best.set_params(early_stopping_rounds = 20)
        print(f"  {modelName} best params: {grid.best_params_}, best CV F1: {grid.best_score_:.4f}")
        return best


class PermutationImportanceAnalyzer:
    def __init__(self) -> None:
        self.m_result = None
        self.m_featureNames = None

    def __call__(self, model, X_test, y_test, featureNames):
        def churnF1Scorer(estimator, X, y):
            return f1_score(y, estimator.predict(X), pos_label = 1, zero_division = 0)

        self.m_featureNames = featureNames
        self.m_result = permutation_importance(
            model, X_test, y_test, scoring = churnF1Scorer,
            n_repeats = 10, random_state = 42,
        )
        return self.m_result

    def plotBar(self, path: str) -> None:
        importances = self.m_result.importances_mean
        stds = self.m_result.importances_std
        indices = numpy.argsort(importances)[::-1][:10]
        topNames = [self.m_featureNames[i] for i in indices]
        topImps = importances[indices]
        topStds = stds[indices]

        pyplot.figure(figsize = (8, 6))
        pyplot.barh(topNames[::-1], topImps[::-1], xerr = topStds[::-1], color = "steelblue")
        pyplot.xlabel("Mean Importance (decrease in F1)")
        pyplot.title("Top 10 Permutation Importances")
        pyplot.savefig(path, dpi = 600, bbox_inches = "tight")
        pyplot.close()


def _computeRocCurve(model, X_test, y_test):
    from sklearn.metrics import roc_curve
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.predict(X_test).astype(float)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return fpr, tpr


if __name__ == "__main__":
    os.makedirs(OUT_PATH, exist_ok = True)

    # --- Load data ---
    loader = DataLoader()
    X_train_raw, y_train, X_test_raw, y_test = loader()
    print(f"Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")
    print(f"Features: {list(X_train_raw.columns)}")
    print(f"Churn rate train: {y_train.mean():.4f}, test: {y_test.mean():.4f}")

    selector = FeatureSelector()
    preprocessor = Preprocessor()
    factory = ModelFactory()
    ablation = AblationConditions()
    evaluator = Evaluator()
    tuner = HyperparamTuner()

    rocData = {}
    allResults = []

    for modelName in MODELS:
        isLR = (modelName == "LR")

        # Feature selection
        X_train_sel = selector(X_train_raw, isLR = isLR)
        X_test_sel  = selector(X_test_raw, isLR = isLR)

        # Preprocessing: fit on train, transform test
        X_train_proc = preprocessor.fitTransform(X_train_sel, isLR = isLR)
        X_test_proc  = preprocessor.transform(X_test_sel, isLR = isLR)

        # Hyperparameter tuning on training set
        print(f"\nTuning {modelName}...")
        tunedModel = tuner(modelName, factory(modelName), X_train_proc, y_train)

        # Ablation conditions
        for condition in CONDITIONS:
            label = f"{modelName}/{condition}"
            print(f"  Fitting {label}...")

            model = ablation(tunedModel, condition, modelName,
                             X_train_proc, y_train, X_test_proc, isLR = isLR)

            result = evaluator(model, X_test_proc, y_test, modelName, condition)
            allResults.append(result)

            # ROC data
            fpr, tpr = _computeRocCurve(model, X_test_proc, y_test)
            rocData[(modelName, condition)] = (fpr, tpr)

            print(f"    {label}: acc={result['accuracy']:.4f}, "
                  f"prec={result['precision']:.4f}, rec={result['recall']:.4f}, "
                  f"f1={result['f1']:.4f}, auc={result['auc_roc']:.4f}")

    # --- Save results ---
    resultsDf = evaluator.saveResultsCsv()
    evaluator.plotRocCurves(allResults, rocData)

    # --- Print results table ---
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print(resultsDf.to_string(index = False))

    # --- Best model by F1 ---
    bestRow = resultsDf.loc[resultsDf["f1"].idxmax()]
    print(f"\nBest model by churn-F1: {bestRow['model']}/{bestRow['condition']} "
          f"(F1={bestRow['f1']:.4f})")

    # --- Permutation importance for best model ---
    # Re-fit best model to get it for permutation importance
    bestModelName = bestRow["model"]
    bestCondition = bestRow["condition"]
    isLR = (bestModelName == "LR")

    X_train_sel = selector(X_train_raw, isLR = isLR)
    X_test_sel  = selector(X_test_raw, isLR = isLR)
    X_train_proc = preprocessor.fitTransform(X_train_sel, isLR = isLR)
    X_test_proc  = preprocessor.transform(X_test_sel, isLR = isLR)

    tunedModel = tuner(bestModelName, factory(bestModelName), X_train_proc, y_train)
    bestModel = ablation(tunedModel, bestCondition, bestModelName,
                         X_train_proc, y_train, X_test_proc, isLR = isLR)

    featureNames = list(X_train_sel.columns) if isLR else FEATURES_10
    permAnalyzer = PermutationImportanceAnalyzer()
    permAnalyzer(bestModel, X_test_proc, y_test, featureNames)
    permAnalyzer.plotBar(os.path.join(OUT_PATH, "permutation_importance.png"))

    print("\nTop 10 Permutation Importances:")
    importances = permAnalyzer.m_result.importances_mean
    indices = numpy.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(indices):
        print(f"  {rank + 1}. {featureNames[idx]}: {importances[idx]:.4f}")

    print(f"\nAll outputs saved to {OUT_PATH}/")