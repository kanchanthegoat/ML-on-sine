'''
We added more powers sine 2 was useless. 3 5 7 9 11 13 27....

The prediction improved with increase in power.. Until we hit power 13.
After that model started to decrease its accuracy and later on started to give garbage result.
We calculated MSE it was also high after 12
This experiment left us with a question. why did it suddenly start to give garbage result after transition from 11 to 13?
Most details are given in main.py
Refer this code for understanding ( dig = degree we are checking)
'''
# code below contains values and modules dependent on main.py. It's only here for understanding not to run.
#If you run it will probably give error. refer to main.py. but understand which part we are changing just in case i forgot to mention in main.py
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree= dig , include_bias=False)
X_poly = poly.fit_transform(x.reshape(-1, 1))
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)
poly_mse = mean_squared_error(y, y_poly_pred)
print("Polynomial model MSE:", poly_mse)
print("Intercept:", poly_model.intercept_)
print("Coefficients:", poly_model.coef_)
