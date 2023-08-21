import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def train(csvFilePath):
    csv_data = pd.read_csv(csvFilePath, sep=',', dtype=np.float64)
    features = csv_data.drop('MEDV', axis=1)
    training_features, testing_features, training_target, testing_target = \
    train_test_split(features, csv_data['MEDV'], random_state=0)

    model = LinearRegression()
    model.fit(training_features, training_target)
    training_pred = model.predict(training_features)
    testing_pred = model.predict(testing_features)

    print('MSE Train : %.3f, Validate : %.3f' % (mean_squared_error(training_target, training_pred), mean_squared_error(testing_target, testing_pred)))
    with open('model.pickle', mode='wb') as f:
        pickle.dump(model, f)


def predict(csvFilePath, picklePath):
    with open(picklePath, mode='rb') as f:
        model = pickle.load(f)

    csv_data = pd.read_csv(csvFilePath, sep=',', dtype=np.float64)
    testing_features = csv_data.drop('MEDV', axis=1)
    testing_target = csv_data['MEDV']
    testing_pred = model.predict(testing_features)
    print('MSE Test : %.3f' % mean_squared_error(testing_target, testing_pred))
    return testing_pred

if __name__ == '__main__':
    train("./boston_train.csv")
    results = predict("./boston_test.csv", "./model.pickle")
    print(results)