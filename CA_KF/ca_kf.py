import import_data as im
import numpy as np

R = np.mat([[10, 0],
            [0, 10]])
H = np.mat([[1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]])
I = np.mat([[1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]])
P_init = Q_init = I
X_prior_ = np.zeros((670, 6))
X_posterior_ = np.zeros((670, 6))
Q_ = np.zeros((670, 6, 6))
P_ = np.zeros((670, 6, 6))

noise_ax = 0.5
noise_ay = 0.5

for i in range(669):
    if i == 0:
        delta_t = 0
        X_prior = np.mat([im.dist_x[i], im.dist_y[i], 0, 0, 0, 0]).T
        X_posterior = X_prior
        P = P_init
        Q = Q_init
    else:
        delta_t = (im.time[i] - im.time[i-1])/1000000
        A = np.mat([[1, 0, delta_t, 0, 1/2 * delta_t ** 2, 0],
                    [0, 1, 0, delta_t, 0, 1/2 * delta_t ** 2],
                    [0, 0, 1, 0, delta_t, 0],
                    [0, 0, 0, 1, 0, delta_t],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])

        # prediction
        X_prior = A * X_posterior
        G = np.mat([[1/6 * delta_t ** 3, 0],
                    [0, 1/6 * delta_t ** 3],
                    [1/2 * delta_t ** 2, 0],
                    [0, 1/2 * delta_t ** 2],
                    [delta_t, 0],
                    [0, delta_t]])
        a_noise = np.mat([[noise_ax, 0],
                          [0, noise_ay]])
        # process covariance
        Q = G * a_noise * G.T
        P = A * P * A.T + Q

        # kalman gain
        K_inver = np.linalg.inv(H * P * H.T + R)
        K = P * H.T * K_inver

        # measurement
        z = np.mat([im.dist_x[i], im.dist_y[i]]).T

        # update state
        X_posterior = X_prior + K * (z - H * X_prior)

        # update covariance
        P = (I - K * H) * P

    # save all the information(prepare for plotting)
    predict = X_prior.T
    X_prior_[i] = predict
    update = X_posterior.T
    X_posterior_[i] = update
    Q_[i] = Q
    P_[i] = P
