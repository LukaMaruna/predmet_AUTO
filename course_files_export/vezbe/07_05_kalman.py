"""
Kalman Filter Demonstration
============================
This example demonstrates how a Kalman filter tracks the position and velocity
of a moving object with noisy measurements.

The Kalman filter is optimal for linear systems with Gaussian noise. It works
in two steps:
1. PREDICT: Estimate where the object will be next based on motion model
2. UPDATE: Correct the prediction using the noisy measurement

Key insight: The filter balances between what we predict (model) and what we
measure (sensors), weighing each by their uncertainty.
"""

from math import *
import matplotlib.pyplot as plt
import numpy as np


class matrix:
    
    # implements basic operations of a matrix class
    
    def __init__(self, value):
        self.value = value
        self.dimx = len(value)
        self.dimy = len(value[0])
        if value == [[]]:
            self.dimx = 0
    
    def zero(self, dimx, dimy):
        # check if valid dimensions
        if dimx < 1 or dimy < 1:
            raise ValueError("Invalid size of matrix")
        else:
            self.dimx = dimx
            self.dimy = dimy
            self.value = [[0 for row in range(dimy)] for col in range(dimx)]
    
    def identity(self, dim):
        # check if valid dimension
        if dim < 1:
            raise ValueError("Invalid size of matrix")
        else:
            self.dimx = dim
            self.dimy = dim
            self.value = [[0 for row in range(dim)] for col in range(dim)]
            for i in range(dim):
                self.value[i][i] = 1
    
    def show(self):
        for i in range(self.dimx):
            print(self.value[i])
        print(' ')
    
    def __add__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx or self.dimy != other.dimy:
            raise ValueError("Matrices must be of equal dimensions to add")
        else:
            # add if correct dimensions
            res = matrix([[]])
            res.zero(self.dimx, self.dimy)
            for i in range(self.dimx):
                for j in range(self.dimy):
                    res.value[i][j] = self.value[i][j] + other.value[i][j]
            return res
    
    def __sub__(self, other):
        # check if correct dimensions
        if self.dimx != other.dimx or self.dimy != other.dimy:
            raise ValueError("Matrices must be of equal dimensions to subtract")
        else:
            # subtract if correct dimensions
            res = matrix([[]])
            res.zero(self.dimx, self.dimy)
            for i in range(self.dimx):
                for j in range(self.dimy):
                    res.value[i][j] = self.value[i][j] - other.value[i][j]
            return res
    
    def __mul__(self, other):
        # check if correct dimensions
        if self.dimy != other.dimx:
            raise ValueError("Matrices must be m*n and n*p to multiply")
        else:
            # multiply if correct dimensions
            res = matrix([[]])
            res.zero(self.dimx, other.dimy)
            for i in range(self.dimx):
                for j in range(other.dimy):
                    for k in range(self.dimy):
                        res.value[i][j] += self.value[i][k] * other.value[k][j]
            return res
    
    def transpose(self):
        # compute transpose
        res = matrix([[]])
        res.zero(self.dimy, self.dimx)
        for i in range(self.dimx):
            for j in range(self.dimy):
                res.value[j][i] = self.value[i][j]
        return res
    
    # Thanks to Ernesto P. Adorio for use of Cholesky and CholeskyInverse functions
    
    def Cholesky(self, ztol=1.0e-5):
        # Computes the upper triangular Cholesky factorization of
        # a positive definite matrix.
        res = matrix([[]])
        res.zero(self.dimx, self.dimx)
        
        for i in range(self.dimx):
            S = sum([(res.value[k][i])**2 for k in range(i)])
            d = self.value[i][i] - S
            if abs(d) < ztol:
                res.value[i][i] = 0.0
            else:
                if d < 0.0:
                    raise ValueError("Matrix not positive-definite")
                res.value[i][i] = sqrt(d)
            for j in range(i+1, self.dimx):
                S = sum([res.value[k][i] * res.value[k][j] for k in range(self.dimx)])
                if abs(S) < ztol:
                    S = 0.0
                try:
                   res.value[i][j] = (self.value[i][j] - S)/res.value[i][i]
                except:
                   raise ValueError("Zero diagonal")
        return res
    
    def CholeskyInverse(self):
        # Computes inverse of matrix given its Cholesky upper Triangular
        # decomposition of matrix.
        res = matrix([[]])
        res.zero(self.dimx, self.dimx)
        
        # Backward step for inverse.
        for j in reversed(range(self.dimx)):
            tjj = self.value[j][j]
            S = sum([self.value[j][k]*res.value[j][k] for k in range(j+1, self.dimx)])
            res.value[j][j] = 1.0/tjj**2 - S/tjj
            for i in reversed(range(j)):
                res.value[j][i] = res.value[i][j] = -sum([self.value[i][k]*res.value[k][j] for k in range(i+1, self.dimx)])/self.value[i][i]
        return res
    
    def inverse(self):
        aux = self.Cholesky()
        res = aux.CholeskyInverse()
        return res
    
    def __repr__(self):
        return repr(self.value)


########################################

# Implement the filter function below

def kalman_filter(x, P):
    """
    Kalman filter that tracks states over time for visualization.

    Returns:
        - Final state (x, P)
        - History of positions, velocities, and uncertainties
    """
    positions = []
    velocities = []
    uncertainties = []

    for n in range(len(measurements)):

        # MEASUREMENT UPDATE (Correction step)
        # We got a new measurement - let's use it to correct our prediction
        Z = matrix([[measurements[n]]])
        y = Z - (H * x)  # Innovation: difference between measurement and prediction
        S = H * P * H.transpose() + R  # Innovation covariance
        K = P * H.transpose() * S.inverse()  # Kalman gain: how much to trust measurement
        x = x + (K * y)  # Corrected state estimate
        P = (I - (K * H)) * P  # Corrected uncertainty

        # Save state after measurement update
        positions.append(x.value[0][0])
        velocities.append(x.value[1][0])
        uncertainties.append(sqrt(P.value[0][0]))  # Position uncertainty (standard deviation)

        # PREDICTION (Time update)
        # Predict where the object will be at the next time step
        x = (F * x) + u  # Predicted state
        P = F * P * F.transpose()  # Predicted uncertainty (grows over time)

    return x, P, positions, velocities, uncertainties

############################################
### Enhanced example with visualization
############################################

# Simulation parameters
dt = 1.0  # Time step (seconds)
num_steps = 50  # Number of time steps
true_velocity = 2.0  # True velocity of the object (m/s)
measurement_noise_std = 4.0  # Standard deviation of measurement noise

# Generate ground truth: object moving with constant velocity
true_positions = [true_velocity * t * dt for t in range(num_steps)]
true_velocities = [true_velocity] * num_steps

# Generate noisy measurements: what our sensor actually sees
np.random.seed(42)  # For reproducibility
measurements = [pos + np.random.normal(0, measurement_noise_std) for pos in true_positions]

# Kalman Filter setup
x = matrix([[0.], [0.]])  # Initial state: [position, velocity]
P = matrix([[500., 0.], [0., 500.]])  # Initial uncertainty (high - we're not sure)
u = matrix([[0.], [0.]])  # No external control input
F = matrix([[1., dt], [0, 1.]])  # State transition: pos_new = pos + vel*dt, vel_new = vel
H = matrix([[1., 0.]])  # Measurement function: we only measure position, not velocity
R = matrix([[measurement_noise_std**2]])  # Measurement noise covariance
I = matrix([[1., 0.], [0., 1.]])  # Identity matrix

# Run the Kalman filter
print("Running Kalman Filter...")
final_x, final_P, filtered_positions, filtered_velocities, uncertainties = kalman_filter(x, P)

print(f"\nFinal estimate:")
print(f"Position: {final_x.value[0][0]:.2f} m (true: {true_positions[-1]:.2f} m)")
print(f"Velocity: {final_x.value[1][0]:.2f} m/s (true: {true_velocity:.2f} m/s)")
print(f"Position uncertainty: ±{sqrt(final_P.value[0][0]):.2f} m")

# Calculate errors
position_errors = [abs(filtered_positions[i] - true_positions[i]) for i in range(len(filtered_positions))]
measurement_errors = [abs(measurements[i] - true_positions[i]) for i in range(len(measurements))]

print(f"\nMean absolute error:")
print(f"Raw measurements: {np.mean(measurement_errors):.2f} m")
print(f"Kalman filter: {np.mean(position_errors):.2f} m")
print(f"Improvement: {(1 - np.mean(position_errors)/np.mean(measurement_errors))*100:.1f}%")

############################################
### Visualization
############################################

time_steps = np.arange(num_steps)

# Create figure with subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Kalman Filter Demonstration: Tracking a Moving Object', fontsize=14, fontweight='bold')

# Plot 1: Position tracking
ax1 = axes[0]
ax1.plot(time_steps, true_positions, 'g-', linewidth=2, label='True Position', alpha=0.8)
ax1.plot(time_steps, measurements, 'r.', markersize=8, label='Noisy Measurements', alpha=0.6)
ax1.plot(time_steps, filtered_positions, 'b-', linewidth=2, label='Kalman Filter Estimate')

# Add uncertainty bounds (±1 standard deviation)
filtered_positions_np = np.array(filtered_positions)
uncertainties_np = np.array(uncertainties)
ax1.fill_between(time_steps,
                  filtered_positions_np - uncertainties_np,
                  filtered_positions_np + uncertainties_np,
                  color='blue', alpha=0.2, label='Uncertainty (±1σ)')

ax1.set_xlabel('Time Step', fontsize=11)
ax1.set_ylabel('Position (m)', fontsize=11)
ax1.set_title('Position Tracking: Kalman Filter vs Raw Measurements', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Velocity estimation
ax2 = axes[1]
ax2.plot(time_steps, true_velocities, 'g-', linewidth=2, label='True Velocity', alpha=0.8)
ax2.plot(time_steps, filtered_velocities, 'b-', linewidth=2, label='Kalman Filter Estimate')
ax2.axhline(y=true_velocity, color='g', linestyle='--', alpha=0.5)
ax2.set_xlabel('Time Step', fontsize=11)
ax2.set_ylabel('Velocity (m/s)', fontsize=11)
ax2.set_title('Velocity Estimation (from position measurements only!)', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Estimation errors
ax3 = axes[2]
ax3.plot(time_steps, measurement_errors, 'r-', linewidth=1.5, label='Raw Measurement Error', alpha=0.6)
ax3.plot(time_steps, position_errors, 'b-', linewidth=2, label='Kalman Filter Error')
ax3.set_xlabel('Time Step', fontsize=11)
ax3.set_ylabel('Absolute Error (m)', fontsize=11)
ax3.set_title('Tracking Error Over Time', fontsize=12)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nKey observations:")
print("1. The Kalman filter smooths noisy measurements (blue line vs red dots)")
print("2. Uncertainty decreases over time as the filter gains confidence")
print("3. The filter estimates velocity even though we only measure position!")
print("4. The filter achieves lower error than raw measurements")