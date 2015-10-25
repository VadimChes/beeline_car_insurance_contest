# Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from load_car_insurance_with_region import load_train_and_test

train_df, y, test_df = load_train_and_test("../data/car_insurance_train.csv", "../data/car_insurance_test.csv")
# create a base classifier used to evaluate a subset of attributes
model = LogisticRegression()
# create the RFE model and select 3 attributes
rfe = RFE(model, 3)
rfe = rfe.fit(train_df, y)
selected_indices = [i for i, x in enumerate(rfe.support_) if x]
print(train_df.iloc[:, selected_indices])
selected_column_names = ['region', 'age']
print(train_df.loc[:, selected_column_names])


