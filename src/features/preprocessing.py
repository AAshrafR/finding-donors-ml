import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocessing_data(data_path):

    data = pd.read_csv(data_path)
    
    # Split target and feature
    income_raw = data['income']
    features_raw = data.drop('income', axis=1)

    #==========================================
    # 1- Outliers and Skewed Features
    #==========================================
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data = features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))


    #==========================================
    # 2- Normalizing
    #==========================================
    scaler = MinMaxScaler() # default=(0, 1)
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])


    #==========================================
    # 3- One-Hot Encoding and Encoding Target
    #==========================================
    # One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
    features_final = pd.get_dummies(features_log_minmax_transform)

    # Encode the 'income_raw' data to numerical values
    income = income_raw.apply(lambda x: 1 if x == '>50K' else 0)

    # Print the number of features after one-hot encoding
    encoded = list(features_final.columns)
    print("{} total features after one-hot encoding.".format(len(encoded)))


    #==========================================
    # 4. Shuffle and Split Data
    #==========================================
    # Split the 'features' and 'income' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                        income, 
                                                        test_size = 0.2, 
                                                        random_state = 0)

    # Show the results of the split
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))
    
    return X_train, X_test, y_train, y_test
