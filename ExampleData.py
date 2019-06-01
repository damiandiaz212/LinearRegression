import matplotlib.pyplot as plt
import pandas as pd
from LinearRegression import gradient_descent

df = pd.read_csv("data_set_example.csv")

X = df["gpa"]
y = df["sat"]


plt.plot(X, y, 'o')
plt.ylabel("SAT Score")
plt.xlabel("GPA (4.0 scale)")


# using linear prediction to predict what sat score given gpa
b, m = gradient_descent(X, y, num_iterations=1000, learning_rate=0.01)
y_predict = [m * x + b for x in X]
plt.plot(X, y_predict, 'o')


plt.show()