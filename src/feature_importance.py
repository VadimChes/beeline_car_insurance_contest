# Feature Importance
from sklearn.ensemble import ExtraTreesClassifier

from load_car_insurance_with_region import load_train_and_test

train_df, y, test_df = load_train_and_test("../data/car_insurance_train.csv", "../data/car_insurance_test.csv")
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(train_df, y)
# display the relative importance of each attribute
print(train_df.columns.values)
print(model.feature_importances_)
