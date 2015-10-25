import re

import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_train_and_test(path_to_train, path_to_test):
    # read data into pandas data frames
    train_df = pd.read_csv(path_to_train,
                           header=0, index_col=0)
    test_df = pd.read_csv(path_to_test,
                          header=0, index_col=0)

    def extract_full_region_number(auto_number):
        match = re.match(r"[A-Z]\d{3}[A-Z]{2}\d{2,3}RUS", auto_number)
        if match:
            index = auto_number.rindex("RUS") - 3
            if auto_number[index] == '1' or auto_number[index] == '7':
                region_number = auto_number[index: index+3]
            else:
                region_number = auto_number[index+1: index+3]
        else:
            region_number = "00"
        return region_number

    def extract_region(auto_number):
        region_number = extract_full_region_number(auto_number)[-2:]

        if region_number == '80':
            region_number = '75'
        elif region_number == '82':
            region_number = '41'
        elif region_number == '93':
            region_number = '23'
        elif region_number in ['84', '88']:
            region_number = '24'
        elif region_number == '81':
            region_number = '59'
        elif region_number == '85':
            region_number = '38'
        elif region_number == '91':
            region_number = '39'
        elif region_number == '90':
            region_number = '50'
        elif region_number == '96':
            region_number = '66'
        elif region_number in ['97', '99']:
            region_number = '77'
        elif region_number == '98':
            region_number = '78'

        return region_number

    def extract_region_orig(auto_number):
        """
        Returns region based on the auto number
        X796TH96RUS -> 96
        E432XX77RUS -> 77
        If there are more than 2-3 digits before 'RUS', returns "not-auto-num"

        """
        index = auto_number.rindex("RUS") - 1
        while auto_number[index].isdigit():
            index -= 1
        auto_number = auto_number[index + 1:auto_number.rindex('RUS')]
        return auto_number if len(auto_number) <= 3 else "not-auto-num"

    def extract_number_type(auto_number):
        num_type = ""
        digits = '0123456789'
        for letter in auto_number[:-3]:
            if letter in digits:
                num_type += '1'
            else:
                num_type += 'a'
        if auto_number[:-3][-3] in '17':
            num_type = num_type[:-1]
        return num_type

    def extract_age(auto_number):
        age = 7
        full_region_number = extract_full_region_number(auto_number)
        if full_region_number in ['102', '113', '116', '121', '80', '82', '93', '84', '81', '125', '126', '134', '136',
                                  '85', '91', '142', '90', '152', '154', '161', '163', '164', '96', '173', '174', '97',
                                  '98', '186']:
            age = 6
        elif full_region_number in ['123', '88', '159', '138', '150', '196', '99', '178']:
            age = 5
        elif full_region_number in ['124', '190', '177']:
            age = 4
        elif full_region_number in ['750', '197']:
            age = 3
        elif full_region_number in ['199']:
            age = 2
        elif full_region_number in ['777']:
            age = 1
        return age

    # auto brand and region are categorical so we encode these columns
    # ex: "Volvo" -> 1, "Audi" -> 2 etc
    auto_brand_encoder = preprocessing.LabelEncoder()
    auto_brand_encoder.fit(train_df['auto_brand'])

    all_auto_numbers = np.append(train_df['auto_number'], test_df['auto_number'])

    regions_train = np.array([extract_region(num) for num in all_auto_numbers])
    region_encoder = preprocessing.LabelEncoder()
    region_encoder.fit(regions_train)

    num_type_train = np.array([extract_number_type(num) for num in train_df['auto_number']])
    num_type_encoder = preprocessing.LabelEncoder()
    num_type_encoder.fit(num_type_train)

    train_df['region'] = region_encoder.transform(train_df['auto_number'].apply(extract_region))
    train_df['num_type'] = num_type_encoder.transform(train_df['auto_number'].apply(extract_number_type))
    train_df['age'] = np.array([extract_age(num) for num in train_df['auto_number']])
    train_df['auto_brand'] = auto_brand_encoder.transform(train_df['auto_brand'])

    # form a numpy array to fit as train set labels
    y = train_df['too_much']
    # we don't need some columns in the training set anymore
    train_df = train_df.drop(['auto_number', 'too_much'], axis=1)

    test_df['region'] = region_encoder.transform(test_df['auto_number'].apply(extract_region))

    test_df['num_type'] = num_type_encoder.transform(test_df['auto_number'].apply(extract_number_type))
    test_df['age'] = np.array([extract_age(num) for num in test_df['auto_number']])
    test_df['auto_brand'] = auto_brand_encoder.transform(test_df['auto_brand'])
    # we don't need some columns in the test set anymore
    test_df = test_df.drop('auto_number', axis=1)
    return train_df, y, test_df

if __name__ == "__main__":
    X_train, y, X_test = load_train_and_test("../data/car_insurance_train.csv",
                                         "../data/car_insurance_test.csv")
    # print(X_train)
    print(X_test)

    predicted_df = pd.DataFrame(X_train)
    predicted_df.to_csv("../output/processed_train.csv")
