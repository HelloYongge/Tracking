import xlrd
import numpy as np
import matplotlib.pyplot as plt
import ctrv_ekf as kf
import import_data as im

xl = xlrd.open_workbook(r'../result.xlsx')
table = xl.sheets()[0]

time = table.col_values(0)
dist_x = table.col_values(1)
dist_y = table.col_values(2)
velocity_x = table.col_values(3)
velocity_y = table.col_values(4)
heading_rad = table.col_values(5)

# 把列表中的时间单位从微秒变换成秒
time = np.divide(im.time, 1000000)
time[:] = time[:] - time[0]

plt.figure(1)
plt.xlabel("time(s)")
plt.ylabel("dist_x(m)")
plt.plot(time-4, dist_x, "k--", label="EKF_CTRV_HJ_Update", linewidth=3)
plt.plot(time, kf.X_posterior_[:, 0], "r--", label="EKF_CTRV_Update", linewidth=3)
plt.plot(time, kf.X_prior_[:, 0], "y--", label="EKF_CTRV_Predict", linewidth=2)
plt.plot(time, im.dist_x, "b--", label="Measurement", linewidth=2)
plt.legend()

plt.figure(2)
plt.xlabel("time(s)")
plt.ylabel("dist_y(m)")
plt.plot(time-4, dist_y, "k--", label="EKF_CTRV_HJ_Update", linewidth=3)
plt.plot(time, kf.X_posterior_[:, 1], "r--", label="EKF_CTRV_Update", linewidth=3)
plt.plot(time, kf.X_prior_[:, 1], "y--", label="EKF_CTRV_Predict", linewidth=2)
plt.plot(time, im.dist_y, "b--",  label="Measurement", linewidth=2)
plt.legend()

plt.figure(3)
plt.xlabel("time(s)")
plt.ylabel("vel(m/s)")
plt.plot(time-4, velocity_x, "k--", label="EKF_CTRV_HJ_Update", linewidth=3)
plt.plot(time, kf.X_posterior_[:, 2], "r--", label="EKF_CTRV_Update", linewidth=3)
plt.plot(time, kf.X_prior_[:, 2], "y--", label="EKF_CTRV_Predict", linewidth=2)
plt.legend()

plt.figure(5)
plt.xlabel("time(s)")
plt.ylabel("heading(rad)")
plt.plot(time-4, heading_rad, "k--", label="EKF_CTRV_HJ_Update", linewidth=3)
plt.plot(time, kf.X_posterior_[:, 3], "r--", label="EKF_CTRV_Update", linewidth=3)
plt.plot(time, kf.X_prior_[:, 3], "y--", label="EKF_CTRV_Predict", linewidth=1)
plt.plot(time, im.heading_rad, "b--", label="Measurement", linewidth=1)
plt.legend()

plt.show()
