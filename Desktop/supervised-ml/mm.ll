import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5],[6]]
y = [20, 30, 40, 50, 65, 76]

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

plt.scatter(X, y, label="Actual marks")
plt.plot(X, predictions, label="Predicted marks")

plt.xlabel("Study Hours")
plt.ylabel("Examination Marks")
plt.title("Linear Regression")
plt.legend()

plt.show()
