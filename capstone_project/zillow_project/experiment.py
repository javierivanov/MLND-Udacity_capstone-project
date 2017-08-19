import gc
from zillow_project.classification_model import ClassificationModel
from zillow_project.mlp_model import MLPRegressionModel
from zillow_project.data import Data, DataSet


if __name__ == "__main__":
    print("Running Regression")
    model = MLPRegressionModel()
    model.run_default("data/LabelEncoder.pkl")
    print("Running Classification")
    model = ClassificationModel()
    model.run_default("data/LabelEncoder.pkl")
