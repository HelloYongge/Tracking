import matplotlib.pyplot as plt
import ca_kf as kf
import import_data as im
import numpy as np

# 把列表中的时间单位从微秒变换成秒
time = np.divide(im.time, 1000000)
time[:] = time[:] - time[0]

plt.figure(1)
plt.xlabel("time(s)")
plt.ylabel("dist_x(m)")
plt.plot(time, kf.X_posterior_[:, 0], "r--", label="KF_CA_Update", linewidth=2)
plt.plot(time, kf.X_prior_[:, 0], "y--", label="KF_CA_Predict", linewidth=2)
plt.plot(time, im.dist_x, "b--", label="Original", linewidth=2)
plt.legend()

plt.figure(2)
plt.xlabel("time(s)")
plt.ylabel("dist_y(m)")
plt.plot(time, kf.X_posterior_[:, 1], "r--", label="KF_CA_Update", linewidth=2)
plt.plot(time, kf.X_prior_[:, 1], "y--", label="KF_CA_Predict", linewidth=2)
plt.plot(time, im.dist_y, "b--",  label="Original", linewidth=2)
plt.legend()

plt.figure(3)
plt.xlabel("time(s)")
plt.ylabel("vel_x(m/s)")
plt.plot(time, kf.X_posterior_[:, 2], "r--", label="KF_CA_Update", linewidth=2)
plt.plot(time, kf.X_prior_[:, 2], "y--", label="KF_CA_Predict", linewidth=2)
plt.legend()

plt.figure(4)
plt.xlabel("time(s)")
plt.ylabel("vel_y(m/s)")
plt.plot(time, kf.X_posterior_[:, 3], "r--", label="KF_CA_Update", linewidth=2)
plt.plot(time, kf.X_prior_[:, 3], "y--", label="KF_CA_Predict", linewidth=2)
plt.legend()

plt.figure(5)
plt.xlabel("time(s)")
plt.ylabel("acc_x(m/s^2)")
plt.plot(time, kf.X_posterior_[:, 4], "r--", label="KF_CA_Update", linewidth=2)
plt.plot(time, kf.X_prior_[:, 4], "y--", label="KF_CA_Predict", linewidth=2)
plt.legend()

plt.figure(6)
plt.xlabel("time(s)")
plt.ylabel("acc_y(m/s^2)")
plt.plot(time, kf.X_posterior_[:, 5], "r--", label="KF_CA_Update", linewidth=2)
plt.plot(time, kf.X_prior_[:, 5], "y--", label="KF_CA_Predict", linewidth=2)
plt.legend()

plt.figure(7)
plt.xlabel("time(s)")
plt.ylabel("diff_x(m)")
plt.plot(time, kf.X_posterior_[:, 0] - kf.X_prior_[:, 0], linewidth=2)
plt.legend()

plt.figure(8)
plt.xlabel("time(s)")
plt.ylabel("diff_y(m)")
plt.plot(time, kf.X_posterior_[:, 1] - kf.X_prior_[:, 1], linewidth=2)
plt.legend()

plt.figure(9)
plt.xlabel("time(s)")
plt.ylabel("diff_vel_x(m/s)")
plt.plot(time, kf.X_posterior_[:, 2] - kf.X_prior_[:, 2], linewidth=2)
plt.legend()

plt.figure(10)
plt.xlabel("time(s)")
plt.ylabel("diff_vel_y(m/s)")
plt.plot(time, kf.X_posterior_[:, 3] - kf.X_prior_[:, 3], linewidth=2)
plt.legend()

# # 观测噪声曲线
# plt.figure(11)
# plt.plot(time, kf.Q_[:, 0, 0])
#
# # 状态协方差的曲线（应该趋向于收敛）-当前状态不对，而且x和y方向的值一样，有点问题
plt.figure(12)
plt.xlabel("time(s)")
plt.ylabel("position_x_covariance")
plt.plot(time, kf.P_[:, 0, 0])
plt.legend()
#
# plt.figure(13)
# plt.plot(time, kf.P_[:, 1, 1])
#
# plt.figure(14)
# plt.plot(time, kf.P_[:, 2, 2])
#
# plt.figure(15)
# plt.plot(time, kf.P_[:, 3, 3])

# 时间步长
time_step = []
for i in range(670):
    if i == 0:
        time_step.append(0.04)
    else:
        time_step.append(time[i] - time[i-1])

plt.figure(16)
plt.xlabel("time(s)")
plt.ylabel("time_step(s)")
plt.plot(time, time_step)
plt.legend()

plt.show()

# 预测值和更新值的差的均值（均值为0是最终的期待，该值越小说明效果越好）
sum_diff = 0.0
for i in range(670):
    diff = kf.X_posterior_[i, 1] - kf.X_prior_[i, 1]
    sum_diff = sum_diff + diff
    print(diff)
print(sum_diff)