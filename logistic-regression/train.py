from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle


def load_data():
    mnist = fetch_openml(data_id=554)

    print(mnist.data.shape, mnist.target.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(mnist.data, mnist.target.astype('int'), test_size=1 / 7.0,
                                                        random_state=0)
    return X_train, X_test, Y_train, Y_test


X_train, X_test, Y_train, Y_test = load_data()

print(X_train.shape, X_test.shape)


def define_model():
    model = LogisticRegression(fit_intercept=True, multi_class='auto', penalty='l1', solver='saga', max_iter=1000, C=50,
                               verbose=2, tol=0.01)
    return model


model = define_model()
model.fit(X_train, Y_train)
score = model.score(X_test, Y_test)
print(score)

pickle.dump(model, open("models/logistic_regression_mnist.pkl", 'wb'))
