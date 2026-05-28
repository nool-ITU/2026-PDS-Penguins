import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV

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
    feature_cols = [c for c in all_feature_cols if c in features_final.columns]
    # Training groups (a–d), testing group (e)
    train_groups = ['a', 'b', 'c', 'd']
    train_df = features_final[features_final["group_id"].isin(train_groups)]
    test_df  = features_final[features_final["group_id"] == "e"]

    X_train = train_df[feature_cols].values
    y_train = train_df["Class"].astype(int).values

    X_test = test_df[feature_cols].values
    y_test = test_df["Class"].astype(int).values
    if load_model:
        model = joblib.load(model_path)
        pass
    else:
        # train the classifier (using logistic regression as an example)
        param_grid = {
            "feature_selector__n_features_to_select": [4, 6, 8, 10, 12],
            "knn__n_neighbors": [3, 5, 7, 9, 11, 15],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"]
        }
        cv = GroupKFold(n_splits=4)

        modelkNN = GridSearchCV(
            estimator=pipe,                 
            param_grid=param_grid,          
            cv=cv,                          
            scoring="f1",                   
            n_jobs=-1,                      
            verbose=2                       
        )
        modelkNN.fit(X_train, y_train, groups=groups_train)
        # save the model.
        joblib.dump(model, model_path)
        pass

    # test the classifier.
    y_prob = model.predict_proba(X_test)[:, 1]

    # write test results to CSV.
    output_df = pd.DataFrame({
        "image_id": test_df["image_id"],
        "true_label": y_test,
        "probability_malignant": y_prob
    })

    output_df.to_csv(prediction_results_path, index=False)


if __name__ == "__main__":
    features_path = "./data/extracted_features_extended_mean.csv"
    prediction_results_path = "./result/predictions/predictions_MODEL.csv"
    model_path = "./result/predictions/predictions_MODEL.csv"
    load_model = False

    main(features_path, prediction_results_path,model_path,load_model)

#features_path for baseline features_path = "./data/extracted_features_baseline_mean.csv"
