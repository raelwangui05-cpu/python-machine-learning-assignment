import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#Features (hours studied)
X = [ [10], [11], [12], [16]]
#labels (test scores)
y = [ 20, 30, 40, 50]
model = LinearRegression()
model.fit(X, y)
#make prediction
hours= 17
prediction = model.predict([[hours]])
print (round(prediction[0], 2))
# Visualize actual data
plt.scatter(X, y, label="Actual marks")

# Visualize the regression line
plt.plot(X, model.predict(X), label="Predicted marks")

plt.xlabel("Study Hours")
plt.ylabel("Examination Marks")
plt.title("Linear Regression: Study Hours Versus Marks")
plt.legend()
plt.show()