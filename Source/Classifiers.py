import os
import numpy
import pandas
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from Encoder import LabelEncoder
from Filter import CorrFilter

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(ROOT_DIR, "Results", "Classification")
IN_TRAIN = os.path.join(ROOT_DIR, "Data", "churn-bigml-80.csv")
IN_TEST = os.path.join(ROOT_DIR, "Data", "churn-bigml-20.csv")

class Classifiers:
    x_train: numpy.ndarray
    y_train: numpy.ndarray 
    x_test: numpy.ndarray
    y_test: numpy.ndarray

    @classmethod
    def samples(cls, x_train: numpy.ndarray, y_train: numpy.ndarray, x_test: numpy.ndarray, y_test: numpy.ndarray):
        cls.x_train = x_train
        cls.y_train = y_train
        cls.x_test = x_test
        cls.y_test = y_test

    def plot(self, classifier: str, y_pred: numpy.ndarray) -> None:
        fpr, tpr, _ = roc_curve(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred)
        pyplot.figure()
        pyplot.xlim([0.0, 1.0])
        pyplot.ylim([0.0, 1.05])
        pyplot.xticks(fontsize=14)
        pyplot.yticks(fontsize=14)
        pyplot.xlabel("False Positive Rate (1-Specificity)", fontsize=16)
        pyplot.ylabel("True Positive Rate (Sensitivity)", fontsize=16)
        line = pyplot.plot(fpr, tpr, marker='o', lw=3, markersize=10, label='ROC curve (area = {:.2f})'.format(roc_auc))[0]
        line.set_clip_on(False)
        pyplot.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
        pyplot.legend(loc="lower right")
        pyplot.tight_layout()
        pyplot.savefig(os.path.join(OUT_PATH, f"{classifier}_ROC.png"), dpi=600, bbox_inches="tight")
        pyplot.close()

    def report(self, classifier: str, y_pred: numpy.ndarray, score: str) -> None:
        print("\n--------------------------------")
        print(f"{classifier} Classification Report:")
        print(f"Wrong classifications rate = {sum(abs(self.y_test - y_pred))/self.x_test.shape[0]}")
        print(f"Classifier score = {score}")
        print(classification_report(self.y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))
        print("--------------------------------\n")


class KNN(Classifiers):
    def tune(self, minN: int = 1, maxN: int = 100) -> None:
        x = []
        f1_scores_c1 = []
        f1_scores_c2 = []
        f1_scores_avg = []

        for k in range(minN, maxN):
            x.append(k)
            classifier = KNeighborsClassifier(n_neighbors=k).fit(self.x_train, self.y_train)
            y_pred = classifier.predict(self.x_test)
            report = classification_report(self.y_test, y_pred, output_dict=True, zero_division=0)
            f1_scores_avg.append(report['macro avg']['f1-score'])
            f1_scores_c1.append(report['0']['f1-score'])
            f1_scores_c2.append(report['1']['f1-score'])
        
        pyplot.figure()
        pyplot.ylim([0.0, 1.05])
        pyplot.xticks(fontsize=14)
        pyplot.yticks(fontsize=14)
        pyplot.xlabel('K', fontsize=16)
        pyplot.ylabel('F1 Score', fontsize=16)
        pyplot.plot(x, f1_scores_c1, label='Class Red', c='#FF0000', lw=2)
        pyplot.plot(x, f1_scores_c2, label='Class Green', c='#00FF00', lw=2)
        pyplot.plot(x, f1_scores_avg, '--', label='macro', c="black", lw=3)
        pyplot.legend(loc="lower left", prop={'size': 14})
        pyplot.tight_layout()
        pyplot.savefig(os.path.join(OUT_PATH, "KNN.png"), dpi=600, bbox_inches="tight")
        pyplot.close()

    def evaluate(self, nNeighbors: int = 5) -> None:
        classifier = KNeighborsClassifier(n_neighbors = nNeighbors)
        classifier.fit(self.x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        self.plot("KNN", y_pred)
        self.report("KNN", y_pred, classifier.score(self.x_test, self.y_test))

class LR(Classifiers):
    def evaluate(self) -> None:
        classifier = LogisticRegression()
        classifier.fit(self.x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        self.plot("LogisticRegression", y_pred)
        self.report("LogisticRegression", y_pred, classifier.score(self.x_test, self.y_test))

class DT(Classifiers):
    def evaluate(self) -> None:
        classifier = DecisionTreeClassifier()
        classifier.fit(self.x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        self.plot("DecisionTree", y_pred)
        self.report("DecisionTree", y_pred, classifier.score(self.x_test, self.y_test))

class RF(Classifiers):
    def evaluate(self) -> None:
        classifier = RandomForestClassifier()
        classifier.fit(self.x_train, self.y_train)
        y_pred = classifier.predict(self.x_test)
        self.plot("RandomForest", y_pred)
        self.report("RandomForest", y_pred, classifier.score(self.x_test, self.y_test))

if __name__ == "__main__":
    os.makedirs(OUT_PATH, exist_ok = True)

    # load and prepare data
    df_train = pandas.read_csv(IN_TRAIN)
    y_train = df_train["Churn"].astype(int).to_numpy()
    x_train = df_train.drop(columns = ["Churn", "State", "Area code"])

    df_test = pandas.read_csv(IN_TEST)
    y_test = df_test["Churn"].astype(int).to_numpy()
    x_test = df_test.drop(columns = ["Churn", "State", "Area code"])

    encoder = LabelEncoder({"Yes": 1, "No": 0})
    x_train = encoder(x_train, "International plan")
    x_train = encoder(x_train, "Voice mail plan")
    x_test = encoder(x_test, "International plan")
    x_test = encoder(x_test, "Voice mail plan")

    filter = CorrFilter()
    x_train = filter(x_train)
    x_test = x_test[x_train.columns] # keep same features as train

    print(f"Features after filtering: {x_train.shape[1]}")
    print(f"Columns: {x_train.columns.tolist()}")

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Classifiers
    Classifiers.samples(x_train, y_train, x_test, y_test)

    # KNN
    knn = KNN()
    # knn.evaluate(minN=2, maxN=5)
    knn.evaluate(nNeighbors=3)
    
    # Logistic Regression
    lr = LR()
    lr.evaluate()

    # Decision Tree
    dt = DT()
    dt.evaluate()

    # Random Forest
    rf = RF()
    rf.evaluate()
