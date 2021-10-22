import matplotlib.pyplot as plt
import ctrv_ekf as kf
import import_data as im
import numpy as np

# 把列表中的时间单位从微秒变换成秒
time = np.divide(im.time, 1000000)
time[:] = time[:] - time[0]

plt.figure(1)
plt.xlabel("time(s)")
plt.ylabel("dist_x(m)")
plt.plot(time, kf.X_posterior_[:, 0], "r--", label="EKF_CTRV_Update", linewidth=3)
plt.plot(time, kf.X_prior_[:, 0], "y--", label="EKF_CTRV_Predict", linewidth=2)
plt.plot(time, im.dist_x, "b--", label="Measurement", linewidth=2)
plt.legend()

plt.figure(2)
plt.xlabel("time(s)")
plt.ylabel("dist_y(m)")
plt.plot(time, kf.X_posterior_[:, 1], "r--", label="EKF_CTRV_Update", linewidth=3)
plt.plot(time, kf.X_prior_[:, 1], "y--", label="EKF_CTRV_Predict", linewidth=2)
plt.plot(time, im.dist_y, "b--",  label="Measurement", linewidth=2)
plt.legend()

plt.figure(3)
plt.xlabel("time(s)")
plt.ylabel("vel(m/s)")
plt.plot(time, kf.X_posterior_[:, 2], "r--", label="EKF_CTRV_Update", linewidth=3)
plt.plot(time, kf.X_prior_[:, 2], "y--", label="EKF_CTRV_Predict", linewidth=2)
plt.legend()

plt.figure(4)
plt.xlabel("time(s)")
plt.ylabel("heading(rad)")
plt.plot(time, kf.X_posterior_[:, 3], "r--", label="EKF_CTRV_Update", linewidth=3)
plt.plot(time, kf.X_prior_[:, 3], "y--", label="EKF_CTRV_Predict", linewidth=2)
plt.plot(time, im.heading_rad, "b--", label="Measurement", linewidth=2)
plt.legend()

# plt.figure(5)
# plt.xlabel("time(s)")
# plt.ylabel("yawrate(rad/s)")
# plt.plot(time, kf.X_posterior_[:, 4], "r--", label="EKF_CTRV_Update", linewidth=2)
# plt.plot(time, kf.X_prior_[:, 4], "y--", label="EKF_CTRV_Predict", linewidth=2)
# plt.legend()
#
# plt.figure(6)
# plt.xlabel("time(s)")
# plt.ylabel("diff_x(m)")
# plt.plot(time, kf.X_posterior_[:, 0] - kf.X_prior_[:, 0], linewidth=2)
# plt.legend()
#
# plt.figure(7)
# plt.xlabel("time(s)")
# plt.ylabel("diff_y(m)")
# plt.plot(time, kf.X_posterior_[:, 1] - kf.X_prior_[:, 1], linewidth=2)
# plt.legend()

plt.show()