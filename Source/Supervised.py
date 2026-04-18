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
                              RocCurveDisplay, precision_recall_curve)
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

FEATURES_6 = [
    "Total day charge",
    "Customer service calls",
    "International plan",
    "Number vmail messages",
    "Total intl charge",
    "Total intl calls",
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


class ThresholdAnalyzer:
    """Sweep decision thresholds for a binary classifier, find optimal points."""

    def __init__(self, model, X_test, y_test) -> None:
        self.m_model = model
        self.m_X_test = X_test
        self.m_y_test = y_test
        self.m_y_proba = None
        self.m_thresholds = None
        self.m_precisions = None
        self.m_recalls = None
        self.m_f1s = None

    def __call__(self, out_path: str) -> None:
        self.m_y_proba = self.m_model.predict_proba(self.m_X_test)[:, 1]
        auc = roc_auc_score(self.m_y_test, self.m_y_proba)

        # Sweep thresholds 0.05 to 0.95 step 0.05
        thresholds = numpy.arange(0.05, 1.0, 0.05)
        precs, recs, f1s = [], [], []
        for t in thresholds:
            preds = (self.m_y_proba >= t).astype(int)
            precs.append(precision_score(self.m_y_test, preds, zero_division = 0.0))
            recs.append(recall_score(self.m_y_test, preds, zero_division = 0.0))
            f1s.append(f1_score(self.m_y_test, preds, zero_division = 0.0))

        self.m_thresholds = thresholds
        self.m_precisions = numpy.array(precs)
        self.m_recalls = numpy.array(recs)
        self.m_f1s = numpy.array(f1s)

        # Find F1-max
        idxF1max = numpy.argmax(self.m_f1s)
        tF1max = thresholds[idxF1max]

        # Find Recall-75: lowest threshold where recall >= 0.75
        # Lower threshold -> higher recall, so first threshold (leftmost) that meets it
        recall75Mask = self.m_recalls >= 0.75
        if recall75Mask.any():
            idxR75 = numpy.where(recall75Mask)[0][0]
        else:
            idxR75 = None

        # Print results
        print(f"\nThreshold Sweep Results:")
        print(f"  F1-max: threshold={tF1max:.2f}, "
              f"precision={self.m_precisions[idxF1max]:.4f}, "
              f"recall={self.m_recalls[idxF1max]:.4f}, "
              f"F1={self.m_f1s[idxF1max]:.4f}")
        if idxR75 is not None:
            tR75 = thresholds[idxR75]
            print(f"  Recall-75: threshold={tR75:.2f}, "
                  f"precision={self.m_precisions[idxR75]:.4f}, "
                  f"recall={self.m_recalls[idxR75]:.4f}, "
                  f"F1={self.m_f1s[idxR75]:.4f}")

        # Plot 1: Threshold sweep
        self._plotSweep(out_path, idxF1max, idxR75)

        # Plot 2: PR curve
        self._plotPRCurve(out_path, auc, tF1max, idxR75)

        # Save thresholds CSV
        self._saveCSV(out_path, auc, idxF1max, idxR75)

    def _plotSweep(self, out_path, idxF1max, idxR75):
        pyplot.figure(figsize = (10, 6))
        pyplot.plot(self.m_thresholds, self.m_precisions, label = "Precision", color = "#1976d2")
        pyplot.plot(self.m_thresholds, self.m_recalls, label = "Recall", color = "#388e3c")
        pyplot.plot(self.m_thresholds, self.m_f1s, label = "F1", color = "#d32f2f")
        pyplot.axvline(x = self.m_thresholds[idxF1max], color = "#d32f2f", linestyle = "--",
                       label = f"F1-max (t={self.m_thresholds[idxF1max]:.2f})")
        if idxR75 is not None:
            pyplot.axvline(x = self.m_thresholds[idxR75], color = "#388e3c", linestyle = "--",
                           label = f"Recall-75 (t={self.m_thresholds[idxR75]:.2f})")
        pyplot.xlabel("Threshold")
        pyplot.ylabel("Score")
        pyplot.title("RF-D: Precision/Recall/F1 vs Threshold")
        pyplot.legend()
        pyplot.tight_layout()
        pyplot.savefig(os.path.join(out_path, "threshold_sweep.png"), dpi = 600, bbox_inches = "tight")
        pyplot.close()

    def _plotPRCurve(self, out_path, auc, tF1max, idxR75):
        precVals, recVals, thrVals = precision_recall_curve(self.m_y_test, self.m_y_proba)
        pyplot.figure(figsize = (8, 6))
        pyplot.plot(recVals, precVals, label = f"PR curve (AUC={auc:.3f})")
        # Default point (threshold ~0.5)
        defaultIdx = numpy.argmin(numpy.abs(thrVals - 0.5))
        pyplot.plot(recVals[defaultIdx], precVals[defaultIdx], "ko",
                    label = "Default (t~0.5)")
        # F1-max point
        f1maxThrIdx = numpy.argmin(numpy.abs(thrVals - tF1max))
        pyplot.plot(recVals[f1maxThrIdx], precVals[f1maxThrIdx], "r^",
                    label = f"F1-max (t={tF1max:.2f})")
        if idxR75 is not None:
            r75Thr = self.m_thresholds[idxR75]
            r75Idx = numpy.argmin(numpy.abs(thrVals - r75Thr))
            pyplot.plot(recVals[r75Idx], precVals[r75Idx], "gs",
                        label = f"Recall-75 (t={r75Thr:.2f})")
        pyplot.xlabel("Recall")
        pyplot.ylabel("Precision")
        pyplot.title("RF-D: Precision-Recall Curve")
        pyplot.legend()
        pyplot.tight_layout()
        pyplot.savefig(os.path.join(out_path, "pr_curve.png"), dpi = 600, bbox_inches = "tight")
        pyplot.close()

    def _saveCSV(self, out_path, auc, idxF1max, idxR75):
        rows = []
        # Default (threshold 0.5)
        defaultIdx = numpy.argmin(numpy.abs(self.m_thresholds - 0.5))
        rows.append({
            "operating_point": "Default (0.5)",
            "threshold": self.m_thresholds[defaultIdx],
            "precision": self.m_precisions[defaultIdx],
            "recall": self.m_recalls[defaultIdx],
            "f1": self.m_f1s[defaultIdx],
            "auc": auc,
        })
        # F1-max
        rows.append({
            "operating_point": "F1-max",
            "threshold": self.m_thresholds[idxF1max],
            "precision": self.m_precisions[idxF1max],
            "recall": self.m_recalls[idxF1max],
            "f1": self.m_f1s[idxF1max],
            "auc": auc,
        })
        # Recall-75
        if idxR75 is not None:
            rows.append({
                "operating_point": "Recall-75",
                "threshold": self.m_thresholds[idxR75],
                "precision": self.m_precisions[idxR75],
                "recall": self.m_recalls[idxR75],
                "f1": self.m_f1s[idxR75],
                "auc": auc,
            })
        df = pandas.DataFrame(rows)
        df.to_csv(os.path.join(out_path, "results_table_thresholds.csv"), index = False)
        print(f"Threshold results saved to {out_path}/results_table_thresholds.csv")


class SixFeatureRunner:
    """Run RF-D on the 6-feature subset and compare with 10-feature baseline."""

    def __init__(self) -> None:
        pass

    def __call__(self, loader, tuner, factory, ablation) -> None:
        # Re-load data using the same DataLoader (already fit on train)
        X_train_raw, y_train, X_test_raw, y_test = loader()

        # Select 6 features only
        X_train_6 = X_train_raw[FEATURES_6]
        X_test_6 = X_test_raw[FEATURES_6]

        print(f"\n6-feature set: {FEATURES_6}")
        print(f"6-feature train shape: {X_train_6.shape}, test shape: {X_test_6.shape}")

        # Preprocess (RF: no scaling, just values)
        preproc6 = Preprocessor()
        X_train_6_proc = preproc6.fitTransform(X_train_6, isLR = False)
        X_test_6_proc = preproc6.transform(X_test_6, isLR = False)

        # Tune RF on 6 features
        print("\nTuning RF on 6 features...")
        tunedRF6 = tuner("RF", factory("RF"), X_train_6_proc, y_train)

        # Fit condition D (calibrated 3-seed ensemble)
        print("Fitting RF/D-6feat...")
        model6 = ablation._fitConditionDEnsemble(tunedRF6, "RF", X_train_6_proc, y_train)

        # Evaluate
        y_pred = model6.predict(X_test_6_proc)
        y_proba = model6.predict_proba(X_test_6_proc)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label = 1, zero_division = 0)
        rec = recall_score(y_test, y_pred, pos_label = 1, zero_division = 0)
        f1 = f1_score(y_test, y_pred, pos_label = 1, zero_division = 0)
        auc = roc_auc_score(y_test, y_proba)

        print(f"\n  RF/D-6feat: acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, "
              f"f1={f1:.4f}, auc={auc:.4f}")

        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plotConfusionMatrix(cm)

        # Append to results_table.csv
        resultRow = {"model": "RF", "condition": "D-6feat",
                     "accuracy": acc, "precision": prec, "recall": rec,
                     "f1": f1, "auc_roc": auc}
        resultsPath = os.path.join(OUT_PATH, "results_table.csv")
        resultsDf = pandas.read_csv(resultsPath)
        resultsDf = pandas.concat([resultsDf, pandas.DataFrame([resultRow])], ignore_index = True)
        resultsDf.to_csv(resultsPath, index = False)
        print(f"Appended RF/D-6feat to {resultsPath}")

        # Print comparison
        tenFeatRow = resultsDf[(resultsDf["model"] == "RF") & (resultsDf["condition"] == "D")]
        if len(tenFeatRow) > 0:
            tenF1 = tenFeatRow["f1"].values[0]
            tenAuc = tenFeatRow["auc_roc"].values[0]
            tenPrec = tenFeatRow["precision"].values[0]
            tenRec = tenFeatRow["recall"].values[0]
            print(f"\n  Comparison: 10-feat RF-D (F1={tenF1:.4f}, AUC={tenAuc:.4f}, "
                  f"prec={tenPrec:.4f}, rec={tenRec:.4f})")
            print(f"               6-feat  RF-D (F1={f1:.4f}, AUC={auc:.4f}, "
                  f"prec={prec:.4f}, rec={rec:.4f})")
            deltaF1 = f1 - tenF1
            print(f"               F1 delta: {deltaF1:+.4f}")

    def _plotConfusionMatrix(self, cm):
        path = os.path.join(OUT_PATH, "confusion_matrix_RF_D_6feat.png")
        pyplot.figure()
        pyplot.imshow(cm, interpolation = "nearest", cmap = pyplot.cm.Blues)
        pyplot.title("Confusion Matrix: RF / D-6feat")
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

    # Save permutation importance as CSV
    permDf = pandas.DataFrame({
        "feature": featureNames,
        "importance_mean": permAnalyzer.m_result.importances_mean,
        "importance_std": permAnalyzer.m_result.importances_std,
    }).sort_values("importance_mean", ascending=False)
    permDf.to_csv(os.path.join(OUT_PATH, "permutation_importance.csv"), index=False)
    print("\nPermutation importance CSV saved.")

    # --- Threshold sweep for best model (RF-D) ---
    import shutil
    shutil.copy2(
        os.path.join(OUT_PATH, "results_table.csv"),
        os.path.join(OUT_PATH, "results_table_before_threshold.csv"))
    print("Backup saved: results_table_before_threshold.csv")

    threshAnalyzer = ThresholdAnalyzer(bestModel, X_test_proc, y_test)
    threshAnalyzer(OUT_PATH)

    # --- 6-feature RF-D comparison ---
    sixFeatRunner = SixFeatureRunner()
    sixFeatRunner(loader, tuner, factory, ablation)

    print("\nTop 10 Permutation Importances:")
    importances = permAnalyzer.m_result.importances_mean
    indices = numpy.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(indices):
        print(f"  {rank + 1}. {featureNames[idx]}: {importances[idx]:.4f}")

    print(f"\nAll outputs saved to {OUT_PATH}/")