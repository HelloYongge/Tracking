import import_data as im
import numpy as np
import math
from math import sqrt
from scipy.linalg import sqrtm
import scipy.linalg as scl
from itertools import chain
import numdifftools as nd
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints

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
n_x = 5
n_x_aug = 7
alpha = 0.9
kapa = 0.2
beta = 2
lamb = alpha ** 2 * (n_x + kapa) - n_x
lamb_aug = alpha ** 2 * (n_x_aug + kapa) - n_x_aug
A = np.zeros((n_x, n_x))
A_aug = np.zeros((n_x_aug, n_x_aug))
Xsigma = np.zeros((n_x, 2 * n_x + 1))
Xsigma_aug = np.zeros((n_x_aug, 2 * n_x_aug + 1))
Xsigma_prior_aug = np.zeros((n_x, 2 * n_x_aug + 1))
Zsigma = np.zeros((3, 2 * n_x_aug + 1))
Zmean = np.zeros(3)
T = np.zeros((5, 3))
S = np.zeros((3, 3))

noise_acc = 0.5
noise_omega = 0.5

for i in range(669):
    if i == 0:
        delta_t = 0
        X_prior = np.array([im.dist_x[i], im.dist_y[i], 0, im.heading_rad[i], 0])
        X_posterior = X_prior
        P = P_init
        Q = Q_init
    else:
        delta_t = (im.time[i] - im.time[i-1])/1000000

        # Build transition_function
        transition_function = lambda y: np.vstack((
            y[0] + (y[2] / y[4]) * (np.sin(y[3] + y[4] * delta_t) - np.sin(y[3]))
            + 1/2 * delta_t ** 2 * y[5] * np.cos(y[3]),
            y[1] + (y[2] / y[4]) * (-np.cos(y[3] + y[4] * delta_t) + np.cos(y[3]))
            + 1/2 * delta_t ** 2 * y[5] * np.sin(y[3]),
            y[2] + 1/2 * y[5] * delta_t ** 3 * y[6],
            y[3] + y[4] * delta_t + 1/2 * delta_t ** 2 * y[6],
            y[4] + delta_t * y[6]
        ))

        # When omega is 0
        transition_function_0 = lambda m: np.vstack((
            m[0] + m[2] * np.cos(m[3]) * delta_t
            + 1/2 * delta_t ** 2 * m[5] * np.cos(m[3]),
            m[1] + m[2] * np.sin(m[3]) * delta_t
            + 1/2 * delta_t ** 2 * m[5] * np.sin(m[3]),
            m[2] + 1/2 * delta_t ** 2 * m[5],
            m[3] + m[4] * delta_t + 1/2 * delta_t ** 2 * m[6],
            m[4] + delta_t * m[6]
        ))

        # GenerateSigmaPoints
        Xsigma[:, 0] = X_posterior
        # Calculate square root of P(A-5*5)
        # A = sqrtm(np.dot((n_x + lamb), P))
        isPositiove = np.all(np.linalg.eigvals(np.dot((n_x + lamb), P)) > 0)
        if(isPositiove):
            A = scl.cholesky(np.dot((n_x + lamb), P))

        for j in range(n_x):
            Xsigma[:, (j+1)] = X_posterior + A[:, j]
            Xsigma[:, (j+1+n_x)] = X_posterior - A[:, j]

        # CalculateSigmaPointWeight
        w_c = 0.5/(n_x + lamb)
        w_m0 = lamb/(n_x + lamb)
        w_c0 = w_m0 + (1 - alpha ** 2 + beta)

        # AugmentedSigmaPoints
        X_aug = np.zeros(7)
        P_aug = np.eye(7)
        X_aug[:5] = X_posterior
        X_aug[5] = 0
        X_aug[6] = 0
        P_aug[0:5, 0:5] = P
        P_aug[5] = noise_acc ** 2
        P_aug[6] = noise_omega ** 2

        Xsigma_aug[:, 0] = X_aug.T
        # Calculate square root of P_aug(A_aug-7*7)

        # A_aug = sqrtm(np.dot((n_x_aug + lamb), P_aug))
        isPositiove = np.all(np.linalg.eigvals(np.dot((n_x_aug + lamb), P_aug)) > 0)
        if(isPositiove):
            A_aug = scl.cholesky(np.dot((n_x_aug + lamb), P_aug))

        for k in range(n_x_aug):
            Xsigma_aug[:, (k+1)] = X_aug.T + A_aug[:, k]
            Xsigma_aug[:, (k+1+n_x_aug)] = X_aug.T - A_aug[:, k]

        # CalculateAugmentedSigmaPointWeight
        w_c_aug = 0.5/(n_x_aug + lamb_aug)
        w_m_aug = w_c_aug
        w_m0_aug = lamb_aug/(n_x_aug + lamb_aug)
        w_c0_aug = w_m0_aug + (1 - alpha ** 2 + beta)

        # SigmaPointPrediction
        for h in range(2 * n_x_aug + 1):
            if np.abs(X_posterior[4]) < 0.0001:
                # squeeze(-1)-压掉最后一个维度
                Xsigma_prior_aug[:, h] = transition_function_0(Xsigma_aug[:, h]).squeeze(-1)
            else:
                Xsigma_prior_aug[:, h] = transition_function(Xsigma_aug[:, h]).squeeze(-1)

        # PredictMeanAndCovariance
        X_prior = w_m0_aug * Xsigma_prior_aug[:, 0]

        for n in range(1, 2 * n_x_aug + 1):
            X_prior += w_m_aug * Xsigma_prior_aug[:, n]

        for g in range(2 * n_x_aug + 1):
            P += w_c_aug * np.dot((Xsigma_prior_aug[:, g] - X_prior),
                                  (Xsigma_prior_aug[:, g] - X_prior).T)

        # measurement
        if (im.heading_rad[i] - im.heading_rad[i-1]) > 0.02:
            z = np.array([[im.dist_x[i]],
                          [im.dist_y[i]],
                          [im.heading_rad[i-1]]])
        else:
            z = np.array([[im.dist_x[i]],
                          [im.dist_y[i]],
                          [im.heading_rad[i]]])

        # Send Predict Point to Measurement Space
        Zsigma = np.dot(H, Xsigma_prior_aug)

        for m in range(2 * n_x_aug + 1):
            # Calculate mean in measurement space
            Zmean += w_m_aug * Zsigma[0:3, m]

        for h in range(2 * n_x_aug + 1):
            # kalman gain
            Xsigma_prior_aug_re = np.reshape(Xsigma_prior_aug[:, m], (Xsigma_prior_aug[:, m].shape[0], 1))
            X_prior_re = np.reshape(X_prior, (X_prior.shape[0], 1))
            Zsigma_re = np.reshape(Zsigma[:, m], (Zsigma[:, m].shape[0], 1))
            Zmean_re = np.reshape(Zmean, (Zmean.shape[0], 1))
            T += w_m_aug * np.dot((Xsigma_prior_aug_re - X_prior_re),
                                  (Zsigma_re - Zmean_re).T)
            S += w_m_aug * np.dot((Zsigma_re - z), (Zsigma_re - Zmean_re).T)

        S += R
        K = np.dot(T, np.linalg.inv(S))
        X_prior = np.reshape(X_prior, (X_prior.shape[0], 1))
        X_posterior = X_prior + np.dot(K, (z - Zmean_re))
        X_posterior = X_posterior.squeeze(-1)
        P = P - np.dot(np.dot(K, S), K.T)

    # # save all the information(prepare for plotting)
    # predict = X_prior.T
    # X_prior_[i] = predict
    # update = X_posterior.T
    # X_posterior_[i] = update
    # Q_[i] = Q
    # P_[i] = P
