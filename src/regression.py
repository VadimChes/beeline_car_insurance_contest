import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn import cross_validation

from load_car_insurance_with_region import load_train_and_test

train_df, y, test_df = load_train_and_test("../data/car_insurance_train.csv", "../data/car_insurance_test.csv")

clf = LogisticRegression()
predicted_labels = cross_validation.cross_val_predict(clf, train_df, y, cv=5)
print("AUC:", roc_auc_score(y, predicted_labels))

clf = clf.fit(train_df, y)

predicted_labels = clf.predict(test_df)
# turn predictions into data frame and save as csv file
predicted_df = pd.DataFrame(predicted_labels, index=np.arange(1, test_df.shape[0] + 1), columns=["too_much"])
predicted_df.to_csv("../output/tree_prediction.csv", index_label="id")

