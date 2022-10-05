import string
import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


class Mylib:
    def preProc_age(age_series, bin=10):
        b = range(15, 91, 5)
        label = list(map(int, range(15, 90, 5)))
        # print(b)
        return pd.cut(age_series, bins=b, labels=label)

    def preProc_eduNum(eduNum_Series):
        b = [1, 3, 6, 9, 11, 13, 30]
        label = ['E', 'M', 'H', 'C', 'U', 'M']
        return pd.cut(eduNum_Series, bins=b, labels=label)


if __name__ == '__main__':
    pass
    # Mylib.preProc_eduNum()


parameters = {'depth': [4, 5, 6, 7, 8, 9, 10],
              'learning_rate': [0.01, 0.02, 0.03, 0.04],
              'iterations': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
              }
Grid_CBC = GridSearchCV(estimator=CBC, param_grid=parameters, cv=2, n_jobs=-1)
Grid_CBC.fit(X_train, y_train)
