from sklearn.metrics import mean_squared_error

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = np.linspace(-np.pi, np.pi, 100)
X = x.reshape(-1, 1)
y = np.sin(x)

for degree in [11, 13]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression(tol=1e-12)
    model.fit(X_poly, y)


    predictions = model.predict(X_poly)
    mse = mean_squared_error(y, predictions)

    print("Training MSE:", mse)
    print("Rank:", model.rank_)

    print(f"\nDegree: {degree}")
    print("Singular values:")
    print(model.singular_)

    threshold = model.singular_[0] * max(X_poly.shape) * np.finfo(float).eps

    print(f"Estimated rank threshold: {threshold:.6e}")
    print(f"Reported rank: {model.rank_}")

'''
    We see that none of the singular values are below the threshold. 
    So, our prediction about the reason being singular value is wrong.
'''


# Check the rank independently
print("NumPy matrix rank:", np.linalg.matrix_rank(X_poly))

# Check the singular values calculated directly from the feature matrix
print("NumPy singular values:", np.linalg.svd(X_poly, compute_uv=False))

# Check the rank reported by scikit-learn
print("Scikit-learn rank:", model.rank_)
print("Scikit-learn singular values:", model.singular_)