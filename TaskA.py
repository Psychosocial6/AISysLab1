import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True):
    if random_state is not None:
        np.random.seed(random_state)
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    split_idx = int(n * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    if hasattr(X, "iloc"):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    else:
        X_train, X_test = X[train_indices], X[test_indices]
    if hasattr(y, "iloc"):
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    else:
        y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X_array = np.array(X)
        self.mean = np.mean(X_array, axis=0)
        self.std = np.std(X_array, axis=0)
        self.std = np.where(self.std == 0, 1e-8, self.std)
        return self

    def transform(self, X):
        X_array = np.array(X)
        return (X_array - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        samples, features = X.shape
        self.weights = np.zeros((features, 1))
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            error = y_predicted - y
            mse_loss = np.mean(error ** 2)
            self.loss_history.append(mse_loss)
            dw = (2 / samples) * np.dot(X.T, error)
            db = (2 / samples) * np.sum(error)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return (np.dot(X, self.weights) + self.bias).flatten()


data = pd.read_csv("day.csv")
drop_cols = ["instant", "dteday", "casual", "registered", "cnt"]
X = data.drop(columns=drop_cols)
y = data["cnt"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = Scaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


linear_regression = LinearRegression(learning_rate=0.05, iterations=100)
linear_regression.fit(X_train_scaled, y_train)

plt.figure(figsize=(8, 5))
plt.plot(linear_regression.loss_history, label="MSE")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()
plt.show()


y_predicted = linear_regression.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_predicted)
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
r2 = r2_score(y_test, y_predicted)


print("Linear Regression")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R^2:   {r2:.4f}")

y_baseline = np.full_like(y_test, fill_value=np.mean(y_train))

b_mae = mean_absolute_error(y_test, y_baseline)
b_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
b_r2 = r2_score(y_test, y_baseline)

print("\nBaseline")
print(f"MAE:  {b_mae:.2f}")
print(f"RMSE: {b_rmse:.2f}")
print(f"R^2:   {b_r2:.4f}")