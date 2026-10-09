from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.linspace(-np.pi, np.pi, 100)
X = x.reshape(-1, 1)

for degree in [11, 13]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, np.sin(x))

    predictions = model.predict(X_poly)
    mse = mean_squared_error(np.sin(x), predictions)

    print(f"\nDegree: {degree}")
    print(f"Training MSE: {mse:.6e}")
    print(f"Model rank: {model.rank_}")
    print(f"Number of features: {X_poly.shape[1]}")
    print(f"Smallest singular value: {model.singular_[-1]:.6e}")

'''
    What we noticed is that the rank has decreased. I do not have enough knowledge here as I'm doing this from scratch
    and I am not able to find any clear documentation stating this exact probelm. I might be first to find out. Nope. Similar cases were discovered before just not exact maybe 
    or I(Gemini & ChatGPT) just didnt search enough.
    This is the stopping point in this program. I will be studying about how everything is calculated in depth.
    And, in next part of 5th experiment we might be able to find out the reason.
'''
# We actually made a very good observation

#OBSERVATION: THE MODEL RANK DECREASED.