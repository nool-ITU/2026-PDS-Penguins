# MODEL FUNCTION

def train_model(features_final, all_feature_cols):
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

    return output_df