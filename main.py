import numpy as np

# Gives 100 equally spaced values from -pi to pi
x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x) # the answer of problem .. since its easily calculatable


from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(x.reshape(-1, 1), y) #This is where training happens, it takes 100 x and y vallues and finds ....
                                # ....the best cofficient for the linear formula y = mx + c


print("Intercept:", model.intercept_) #c value
print("Slope:", model.coef_[0]) #m value

y_pred = model.predict(x.reshape(-1, 1)) # checking the output of training

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y, y_pred) # calculating how good our model is:
print("Linear model MSE:", mse)

#Both curves in plotting
import matplotlib.pyplot as plt
plt.plot(x, y, label="True sin(x)")
plt.plot(x, y_pred, label="Linear model")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

'''After looking at this graph, I realized that linear method we 
     are trying can not get anywhere close to getting sinx right.
     It is very obvious but for some reasoon I thought it would get close 
     to it. It ofcourse can't. So, is this the limitation of ML? :(
'''


'''And another thing, the line is closest we can get to sinx curve using this method. 
    So, we can say that this is the best linear approximation of sinx curve. 
    So, My project is over right?? just 20 lines of code and 30 lines of comments 
    Our model was actually a very good model, it literally gives us the best linear approximation of sinx curve.  for the given range of x.
    ..if we increase the range of x to more than -pi to pi then the function is just x axis(best linear approximation
    of sine and cosine curves).), 
    Hence, this model is a good model. But, we can not use this method to get sinx curve.
'''

# Hence, I conclude that linear regression is not a good method to get  sinx curve:

'''
Experiment 1: Linear Regression on sin(x)

Research question:
Can a linear model approximate the sine function?

Result:
Intercept: approximately 0
Slope: 0.294866
MSE: 0.20318

Observation:
The model found the best-fitting straight line, but it
could not capture the curved shape of the sine function.

Conclusion:
The limitation is the linear model's expressive capacity,
not simply a failure to optimize its parameters.

Next question:
Can a more expressive model learn the sine curve better?
'''


from sklearn.preprocessing import PolynomialFeatures
# Previously, we passed x only, Now, we will pass x and x^2. which will range from 0 to 2*pi.
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(x.reshape(-1, 1))

# Training the linear regression in quadratic style.
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Calculating the coefficients. a,b and c of ax + by + c = 0
y_poly_pred = poly_model.predict(X_poly)

# Measuring the error in this style of solving.
poly_mse = mean_squared_error(y, y_poly_pred)
print("Polynomial model MSE:", poly_mse) # Got the same MSE :o

# Lets see the coefficients to find out the reason.
print("Intercept:", poly_model.intercept_)
print("Coefficients:", poly_model.coef_)

''' 
    So, it just gave us same equation as linear regression.
    Adding x^2 did not help at all. It only used more electricity and made my laptop little older.

    Let's wrap this up:
    (Experiment 2: Polynomial Regression on sin(x)

    Queation: does adding x^2 help to improve linear model's performance on sin(x) approximation?
    Result: Both models have same MSE and the equation.
    Observation: The coefficient of x^2 is 0. Thanks to the symmetry of the sine function. 
    conclusion: Adding a feature does not gurantee improvement.

    Next: Will adding x^3 help? Let's find out.
    
    To do this I only need to change one character in the code I wrote  degree=3 instead of 2. 
'''

# Same exact code , but with degree 3.
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(x.reshape(-1, 1))
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)
poly_mse = mean_squared_error(y, y_poly_pred)
print("Polynomial model MSE:", poly_mse)
print("Intercept:", poly_model.intercept_)
print("Coefficients:", poly_model.coef_)


'''
    We can see it gave way better result when we considered x^3. It's almost a sine curve. The MSE is also very low.
    So, for experiment 3: I conclude that sine x can be approximated using polynimal regression. ( for the given range minus pi to pi)
    But I can see the end of the graph growing too quickly.
    So, for the experiment 4. I'm going to see if this sinx  curve is being approximated out of range too.
'''
# Just simply plotting the prediction for -2pi to +2pi..
x_test = np.linspace(-2 * np.pi, 2 * np.pi, 300)
y_test = np.sin(x_test)
y_linear_test = model.predict(x_test.reshape(-1, 1))
X_test_poly = poly.transform(x_test.reshape(-1, 1))
y_poly_test = poly_model.predict(X_test_poly)
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, label="True sin(x)")
plt.plot(x_test, y_linear_test, label="Linear model")
plt.plot(x_test, y_poly_test, label="Cubic model")
plt.axvline(-np.pi, linestyle="--", color="gray")
plt.axvline(np.pi, linestyle="--", color="gray")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Interpolation vs Extrapolation")
plt.legend()
plt.grid(True)
plt.show()

# Keep only test points outside the training range
outside = (x_test < -np.pi) | (x_test > np.pi)

# Calculate MSE only on those outside points
outside_mse = mean_squared_error(
    y_test[outside],
    y_poly_test[outside]
)

print("Extrapolation MSE:", outside_mse)




'''


The lines below are only checking the cause of extreme values after power 13...
It shall be run only after changing the degree to 27. If you did not it just gives curve for the power 3 falsely telling that it is for power 27.
which might be confusing. I haven't commented it because it is very crucial part of the experiment.

'''


plt.figure(figsize=(10, 5))

plt.plot(x, y, label="True sin(x)")
plt.plot(x, y_poly_pred, label="Degree 27 prediction")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Degree 27 — Training Range Only")
plt.legend()
plt.grid(True)
plt.show()

print("Training prediction min:", np.min(y_poly_pred))
print("Training prediction max:", np.max(y_poly_pred))

print("Outside prediction min:", np.min(y_poly_test[outside]))
print("Outside prediction max:", np.max(y_poly_test[outside]))

print("Training prediction min:", np.min(y_poly_pred))
print("Training prediction max:", np.max(y_poly_pred))

print("Outside prediction min:", np.min(y_poly_test[outside]))
print("Outside prediction max:", np.max(y_poly_test[outside]))



# Experiment 4: Compare high polynomial degrees

degrees = [5, 7, 9, 11, 13, 27]

for degree in degrees:
    # Create polynomial features for this degree
    poly_test = PolynomialFeatures(
        degree=degree,
        include_bias=False
    )

    X_train_poly = poly_test.fit_transform(x.reshape(-1, 1))
    X_test_poly = poly_test.transform(x_test.reshape(-1, 1))

    # Train a fresh model for this degree
    model_test = LinearRegression()
    model_test.fit(X_train_poly, y)

    # Calculate training and extrapolation predictions
    train_pred = model_test.predict(X_train_poly)
    test_pred = model_test.predict(X_test_poly)

    # Measure both errors
    train_mse = mean_squared_error(y, train_pred)
    extrapolation_mse = mean_squared_error(
        y_test[outside],
        test_pred[outside]
    )

    print(f"\nDegree: {degree}")
    print(f"Training MSE: {train_mse:.6e}")
    print(f"Extrapolation MSE: {extrapolation_mse:.6e}")


# Experiment 5: Investigating high-degree feature scales

for degree in [3, 11, 13, 27]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(x.reshape(-1, 1))

    print(f"\nDegree: {degree}")
    print("Smallest feature maximum:", np.min(np.max(np.abs(X_poly), axis=0)))
    print("Largest feature maximum:", np.max(np.max(np.abs(X_poly), axis=0)))