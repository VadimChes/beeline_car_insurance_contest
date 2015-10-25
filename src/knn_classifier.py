import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

from load_car_insurance_with_region import load_train_and_test

if __name__ == "__main__":
    # read data
    train_df, y, test_df = load_train_and_test("../data/car_insurance_train.csv", "../data/car_insurance_test.csv")

    use_features = ['compensated', 'region', 'auto_brand', 'num_type']
    train_df = train_df.loc[:, use_features]
    test_df = test_df.loc[:, use_features]

    knn_params = {'n_neighbors': list(range(2, 100, 1))}

    locally_best_tree = GridSearchCV(KNeighborsClassifier(),
                                     knn_params,
                                     verbose=True, n_jobs=4, cv=5,
                                     scoring="roc_auc")
    locally_best_tree.fit(train_df, y)
    print("Best params:", locally_best_tree.best_params_)
    print("Best cross validaton score", locally_best_tree.best_score_)

    # make predictions. this one is worth 0.735 ROC AUC
    predicted_labels = locally_best_tree.best_estimator_.predict(test_df)

    # turn predictions into data frame and save as csv file
    predicted_df = pd.DataFrame(predicted_labels,
                                index = np.arange(1, test_df.shape[0] + 1),
                                columns=["too_much"])
    predicted_df.to_csv("../output/knn_prediction.csv", index_label="id")