from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

columns = list()
for i in range(0, 8000):
    columns.append(i)
columns_10 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

X = pd.read_csv("../data/Vecv-1000.csv", names=columns)
Y = pd.read_csv("../data/thetav-1000.csv", names=columns_10)

sc = MinMaxScaler()
X = pd.DataFrame(sc.fit_transform(X))
sc2 = MinMaxScaler()
Y = pd.DataFrame(sc2.fit_transform(Y))

def svr_process(param, epsilon=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y.iloc[:, param], train_size=0.70, test_size=0.30, random_state=101)
    model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=epsilon))
    y_pred = model.fit(X_train, y_train).predict(X_test)

    plt.plot(X_test.iloc[:, 0], y_pred, 'bo', color='red', label='Predição')
    plt.scatter(X_test.iloc[:, 0], y_test, s=5, color='blue', label='Original')
    plt.legend()
    plt.show()

    score = model.score(X_train, y_train)
    print("R-squared: ", score)
    print("MSE: ", mean_squared_error(y_test, y_pred))

svr_process(1, 0)