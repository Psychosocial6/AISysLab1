import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, log_loss, confusion_matrix
)

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



class LogisticRegression:
    def __init__(self, learning_rate=0.1, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def compute_loss(self, y_true, y_predicted_prob):
        eps = 1e-15
        y_predicted_prob = np.clip(y_predicted_prob, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(y_predicted_prob) + (1.0 - y_true) * np.log(1.0 - y_predicted_prob))

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float).reshape(-1, 1)
        samples, features = X.shape

        self.weights = np.zeros((features, 1))
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted_prob = self.sigmoid(linear_model)
            loss = self.compute_loss(y, y_predicted_prob)
            self.loss_history.append(loss)

            dw = (1 / samples) * np.dot(X.T, (y_predicted_prob - y))
            db = (1 / samples) * np.sum(y_predicted_prob - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_prob(self, X):
        X = np.array(X, dtype=float)
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model).flatten()

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_prob(X)
        return (probabilities >= threshold).astype(int)

data = pd.read_csv("train.csv")
drop_cols = ["PassengerId", "Name", "Cabin"]
df_clean = data.drop(columns=drop_cols).copy()
num_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
cat_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP"]

for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in cat_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

y = df_clean["Transported"].astype(int)
X = df_clean.drop(columns=["Transported"])
X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = Scaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


log_reg = LogisticRegression(learning_rate=0.1, iterations=1000)
log_reg.fit(X_train_scaled, y_train)
plt.figure(figsize=(8, 5))
plt.plot(log_reg.loss_history, color="orange", label="Train LogLoss")
plt.title("Logistic Regression")
plt.xlabel("Epoch")
plt.ylabel("LogLoss")
plt.grid(True)
plt.legend()
plt.show()

y_predicted_prob = log_reg.predict_prob(X_test_scaled)
y_predicted = log_reg.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_predicted)
precision = precision_score(y_test, y_predicted)
recall = recall_score(y_test, y_predicted)
f1 = f1_score(y_test, y_predicted)
loss = log_loss(y_test, y_predicted_prob)

print("Logistic Regression")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"LogLoss: {loss:.4f}")
cm = confusion_matrix(y_test, y_predicted)
tn, fp, fn, tp = cm.ravel()
print("\nConfusion Matrix")
print(f"True Negative (TN):  {tn} | False Positive (FP): {fp}")
print(f"False Negative (FN): {fn} | True Positive (TP):  {tp}")

majority_class = y_train.mode()[0]
y_baseline = np.full_like(y_test, fill_value=majority_class)
majority_prob = np.mean(y_train == majority_class)
y_baseline_prob = np.full_like(y_test, fill_value=majority_prob, dtype=float)

b_accuracy = accuracy_score(y_test, y_baseline)
b_precision = precision_score(y_test, y_baseline, zero_division=0)
b_recall = recall_score(y_test, y_baseline, zero_division=0)
b_f1 = f1_score(y_test, y_baseline, zero_division=0)
b_loss = log_loss(y_test, y_baseline_prob)

print("\nBaseline")
print(f"Maj class {majority_class}")
print(f"Accuracy: {b_accuracy:.4f}")
print(f"Precision: {b_precision:.4f}")
print(f"Recall: {b_recall:.4f}")
print(f"F1-Score: {b_f1:.4f}")
print(f"LogLoss: {b_loss:.4f}")