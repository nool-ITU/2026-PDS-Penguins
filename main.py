import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from temp_code.train_model import train_model

def main(features_path, prediction_results_path, model_path, load_model):
    """
    Docstring for main
    
    :param features_path: Path to the features csv used as input to the model (e.g. ./data/features.csv).
    :param prediction_results_path: Path to save the output predictions of the model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param model_path: Path to save or load the trained model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param load_model: Boolean to train the model and save it to model_path if False, load it from model_path if True. 
    """
    
    # load dataset CSV file
    features_final= pd.read_csv(features_path)
    # split the dataset into training and testing sets.
    #our csv already have them because we applayed split_data_k_fold.py before
    all_feature_cols = [
        "Asymmetry", "Compactness", "Convexity", "Multicolor",
        "red_pixels", "blue_pixels", "mean_red", "mean_blue",
        "Entropy", "Hue_Var", "Sat_Var", "Val_Var", "hair_feature"]
    output_df = train_model(features_final, all_feature_cols, )

    output_df.to_csv(prediction_results_path, index=False)


if __name__ == "__main__":
    features_path = "./data/extracted_features_extended_mean.csv"
    prediction_results_path = "./result/predictions/predictions_MODEL.csv"
    model_path = "./result/predictions/predictions_MODEL.csv"
    load_model = False

    main(features_path, prediction_results_path,model_path,load_model)

#features_path for baseline features_path = "./data/extracted_features_baseline_mean.csv"
