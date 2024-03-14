import matplotlib.pyplot as plt
import numpy as np

# Utility functions for player A and B
def u_A(x_A1, x_A2):
    return x_A1**(1/3)* x_A2**(1-(1/3))

def u_B(x_B1, x_B2):
    return x_B1**(2/3) * x_B2**(1-(2/3))

# Initial endowment
omega_1A = 0.8
omega_2A = 0.3
omega_1B = 1 - omega_1A
omega_2B = 1 - omega_2A

# Create Edgeworth box
fig, ax = plt.subplots()

# Plot initial endowment point
ax.scatter(omega_1A, omega_2A, marker='o', color='red', label='Initial Endowment (B)')

# Plot indifference curves
# xA1_values = np.linspace(0, 1, 100)
# xA2_values = np.linspace(0, 1, 100)
# X, Y = np.meshgrid(xA1_values, xA2_values)
# 
# Z_A = u_A(X, Y)
# Z_B = u_B(1 - X, 1 - Y)
# 
# contour_A = ax.contour(X, Y, Z_A, levels=10, colors='red', label='Indifference Curve (A)')
# contour_B = ax.contour(X, Y, Z_B, levels=10, colors='blue', label='Indifference Curve (B)')

# Find Pareto improvements relative to the endowment
N = 75
xA1_values = np.linspace(0, 1, N + 1)
xA2_values = np.linspace(0, 1, N + 1)

for xA1 in xA1_values:
    for xA2 in xA2_values:
        x1 = 1 - xA1
        x2 = 1 - xA2

        if u_A(xA1, xA2) >= u_A(omega_1A, omega_2A) and u_B(x1, x2) >= u_B(omega_1B, omega_2B):
            ax.scatter(xA1, xA2, color='green', marker='o')

# Set labels and legend
ax.set_xlabel('$x_{A1}$')
ax.set_ylabel('$x_{A2}$')
ax.set_title('Edgeworth Box with Pareto Improvements')
ax.legend()

plt.show()