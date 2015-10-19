import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from load_car_insurance_with_region import load_train_and_test

# read data
train_df, y, test_df = load_train_and_test("../data/car_insurance_train.csv",
                                         "../data/car_insurance_test.csv")
# parameter combinations to try
tree_params = {'criterion': ('gini', 'entropy'),
               'max_depth': list(range(1,11)),
               'min_samples_leaf': list(range(1,11))}

locally_best_tree = GridSearchCV(DecisionTreeClassifier(),
                                 tree_params,
                                 verbose=True, n_jobs=4, cv=5,
                                 scoring="roc_auc")
locally_best_tree.fit(train_df, y)

print("Best params:", locally_best_tree.best_params_)
print("Best cross validaton score", locally_best_tree.best_score_)

# export tree visualization
# after that $ dot -Tpng tree.dot -o tree.png    (PNG format)
# or open in Graphviz
export_graphviz(locally_best_tree.best_estimator_, out_file="../output/tree.dot")

# make predictions. this one is worth 0.735 ROC AUC
predicted_labels = locally_best_tree.best_estimator_.predict(test_df)

# turn predictions into data frame and save as csv file
predicted_df = pd.DataFrame(predicted_labels,
                            index = np.arange(1, test_df.shape[0] + 1),
                            columns=["too_much"])
predicted_df.to_csv("../output/tree_prediction.csv", index_label="id")