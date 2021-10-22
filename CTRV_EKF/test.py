import import_data as im
import numpy as np
import numdifftools as nd

R = np.array([[10, 0, 0],
              [0, 10, 0],
              [0, 0, 0.05]])
H = np.array([[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0]])
I = np.eye(5)
P_init = Q_init = I
X_prior_ = np.zeros((670, 5))
X_posterior_ = np.zeros((670, 5))
Q_ = np.zeros((670, 5, 5))
P_ = np.zeros((670, 5, 5))

noise_acc = 0.5
noise_omega = 0.5

delta_t = 0.04
P = P_init
Q = Q_init

# Build transition_function
transition_function = lambda y: np.vstack((
    y[0] + (y[2] / y[4]) * (np.sin(y[3] + y[4] * delta_t) - np.sin(y[3])),
    y[1] + (y[2] / y[4]) * (-np.cos(y[3] + y[4] * delta_t) + np.cos(y[3])),
    y[2],
    y[3] + y[4] * delta_t,
    y[4]
))

# When omega is 0
transition_function_0 = lambda m: np.vstack((
    m[0] + m[2] * np.cos(m[3]) * delta_t,
    m[1] + m[2] * np.sin(m[3]) * delta_t,
    m[2],
    m[3] + m[4] * delta_t,
    m[4]
))

# Calculate Jacobian
J_A = nd.Jacobian(transition_function)
J_A_0 = nd.Jacobian(transition_function_0)

X_posterior = np.array([290.565513063, -3.313665807, 0.000904366, -0.031266593, 0.000077655])

# Prediction
if np.abs(X_posterior[4]) < 0.0001:
    X_prior = transition_function_0(X_posterior)
    JA = J_A_0(X_posterior)
else:
    X_prior = transition_function(X_posterior)
    JA = J_A(X_posterior)

# Process Covariance
G = np.zeros([5, 2])
G[0, 0] = 0.5 * delta_t * delta_t * np.cos(X_posterior[3])
G[1, 0] = 0.5 * delta_t * delta_t * np.sin(X_posterior[3])
G[2, 0] = delta_t
G[3, 1] = 0.5 * delta_t * delta_t
G[4, 1] = delta_t

Q_v = np.diag([noise_acc * noise_acc, noise_omega * noise_omega])
Q = np.dot(np.dot(G, Q_v), G.T)

# State Covariance
P = np.dot(np.dot(JA, P), JA.T) + Q

# kalman gain
S = np.dot(np.dot(H, P), H.T) + R
K = np.dot(np.dot(P, H.T), np.linalg.inv(S))

# measurement
z = np.array([[290.670013428], [-3.673799992], [-0.031216800]])

# Update State
y = z - np.dot(H, X_prior)
X_posterior = X_prior + np.dot(K, y)

# Update State Covariance
P = np.dot((I - np.dot(K, H)), P)

print(X_prior)
print(X_posterior)