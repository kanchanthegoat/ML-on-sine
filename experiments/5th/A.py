#COEFFICIENT INFORMATION
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x)
X = x.reshape(-1, 1)

for degree in [11, 13, 15, 20, 27]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    predictions = model.predict(X_poly)
    mse = mean_squared_error(y, predictions)

    print(f"\nDegree: {degree}")
    print(f"Training MSE: {mse:.6e}")
    print(f"Largest absolute coefficient: {np.max(np.abs(model.coef_)):.6e}")
    print(f"Smallest absolute coefficient: {np.min(np.abs(model.coef_)):.6e}")
    print("Coefficients:")
    print(model.coef_)

'''
    We can observe that coefficiets are going rapidly low after 11. Could that be the reason??
    I suspect that. But I am not sure.
    Even if it is, we still can't get to the conclusion as the coefficient are going low for a reason.
    We can not conclude anything from this part. 
    We only observed the coefficients in this file. Nothing more...
    The experiment 5 was a very unexpected experiment. 
    This was not part of my roadmap. 
    But, I had to do this because I encountered weird behaviour after power 11.
    Now, we need to find the reason anyhow. This one is going to be one of the longest and exciting (for me) experiment.
'''
