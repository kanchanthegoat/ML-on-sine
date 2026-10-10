
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev

# Experiment 6: Approximate sin(x) over a wide interval.
# Training: [-10*pi, 10*pi]
# Plot:     [-20*pi, 20*pi]

TRAIN_LIMIT = 10 * np.pi
PLOT_LIMIT = 20 * np.pi
DEGREE = 50
N_TRAIN = 5000

# Dense, evenly spaced training data
x_train = np.linspace(-TRAIN_LIMIT, TRAIN_LIMIT, N_TRAIN)
y_train = np.sin(x_train)

# Chebyshev.fit scales the input interval internally and uses a
# Chebyshev basis instead of raw x, x^2, x^3, ... features.
model = Chebyshev.fit(
    x_train,
    y_train,
    deg=DEGREE,
    domain=[-TRAIN_LIMIT, TRAIN_LIMIT]
)

# Evaluate and validate the fit on independent points
x_test = np.linspace(-TRAIN_LIMIT, TRAIN_LIMIT, 10001)
y_true = np.sin(x_test)
y_pred = model(x_test)

train_rmse = np.sqrt(np.mean((model(x_train) - y_train) ** 2))
test_rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
max_test_error = np.max(np.abs(y_pred - y_true))

print(f"Polynomial degree: {DEGREE}")
print(f"Training RMSE:     {train_rmse:.12e}")
print(f"Test RMSE:         {test_rmse:.12e}")
print(f"Maximum test error:{max_test_error:.12e}")

# Compare the learned polynomial against sin(x) on a wider interval
x_plot = np.linspace(-PLOT_LIMIT, PLOT_LIMIT, 6000)
y_actual = np.sin(x_plot)
y_model = model(x_plot)

plt.figure(figsize=(13, 6))
plt.plot(x_plot / np.pi, y_actual, label="True sin(x)", linewidth=2)
plt.plot(x_plot / np.pi, y_model, "--", label="Chebyshev polynomial", linewidth=1.5)
plt.axvline(-10, color="gray", linestyle=":", label="Training boundaries")
plt.axvline(10, color="gray", linestyle=":")
plt.xlabel("x / pi")
plt.ylabel("y")
plt.title("Experiment 6: Polynomial approximation of sin(x)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Inspect the error across the entire plotted interval
error = np.abs(y_model - y_actual)

plt.figure(figsize=(13, 4))
plt.plot(x_plot / np.pi, error)
plt.axvline(-10, color="gray", linestyle=":")
plt.axvline(10, color="gray", linestyle=":")
plt.xlabel("x / pi")
plt.ylabel("Absolute error")
plt.title("Approximation error, including extrapolation")
plt.grid(True)
plt.tight_layout()
plt.show()

# User query: enter 67.5*pi, 100, or any other value in radians.
while True:
    entry = input("\nEnter x in radians (or q to quit): ").strip()

    if entry.lower() == "q":
        break

    try:
        x_value = float(entry)
        prediction = float(model(x_value))
        actual = float(np.sin(x_value))

        print(f"Polynomial prediction: {prediction:.15g}")
        print(f"NumPy sin(x):          {actual:.15g}")
        print(f"Absolute error:        {abs(prediction - actual):.6e}")

        if abs(x_value) > TRAIN_LIMIT:
            print("Note: this input is outside the training interval.")

    except ValueError:
        print("Please enter a valid number in radians, or q to quit.")