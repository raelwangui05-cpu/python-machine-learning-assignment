from sklearn.linear_model import LinearRegression

#Features(hours studied)
X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

#Target variable (Exam scores)
y = [30, 40, 45, 55, 65, 70, 80, 85, 90, 95]

#Create model
model = LinearRegression()

#Train the model
model.fit(X, y)

#Allow user to input hours studied
hours_studied = float(input("Enter the number of hours studied: "))

#Make prediction
prediction = model.predict([[hours_studied]])

print(round(prediction[0], 2))

#visualization
import matplotlib.pyplot as plt

#Plotting the data points
plt.scatter(X, y, color='blue', label='Data points')

#Plotting the regression line
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.title('Hours Studied vs Exam Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Scores')
plt.legend()
plt.show()