import csv
import matplotlib.pyplot as plt
import numpy as np


def find_the_intersection(ego, target):
    # 确保ego的数据最小，后面插值才没问题
    if ego[0, 0] < target[0, 0]:
        for i in range(len(ego[0])):
            if ego[0, i] < target[0, 0] < ego[0, i+1]:
                ego_start = i + 1
                target_start = 0
                break
            else:
                continue
    else:
        for i in range(len(target[0])):
            if target[0: i] < ego[0, 0] < target[0, i+1]:
                ego_start = 0
                target_start = i
                break
            else:
                continue
    if ego[0, -1] < target[0, -1]:
        for i in range(len(target[0])):
            if target[0, i] < ego[0, -1] < target[0, i+1]:
                ego_end = len(ego[0]) - 1
                target_end = i + 1
            else:
                continue
    else:
        for i in range(len(ego[0])):
            if ego[0, i] < target[0, -1] < ego[0, i+1]:
                ego_end = i
                target_end = len(target[0]) - 1
            else:
                continue
    index = [ego_start, ego_end, target_start, target_end]
    return index


def reshape(original, camera):
    for i in range(len(original[0])):
        if original[0, i] < camera[0, 0]:
            continue
        else:
            start = i
            break
    for i in range(len(original[0])):
        if original[0, i] < camera[0, len(camera[0]) - 1]:
            continue
        else:
            end = i
            break
    new = original[0:(original.shape[0]), start:end]
    return new


def match(ego, target):
    new = []
    for i in range(len(ego[0])):
        for j in range(len(target[0])):
            if target[0, j] < ego[0, i] < target[0, j + 1]:
                new.append((target[:, j + 1] - target[:, j]) / (target[0, j + 1] - target[0, j]) * (
                            ego[0, i] - target[0, j]) + target[:, j])
                break
            else:
                continue
    target_match = np.concatenate([a.reshape(-1, 1) for a in new], axis=1)
    return target_match


def calculate_rel(ego, target):
    # 以匹配好的数据为基准，计算相对结果
    new = []
    for i in range(len(target[0])):
        for j in range(len(ego[0])):
            if ego[0, j] == target[0, i]:
                new.append(target[:, i] - ego[:, j])
                break
            else:
                continue
    target_rel = np.concatenate([a.reshape(-1, 1) for a in new], axis=1)
    target_rel[0, :] = target[0, :]
    return target_rel


def convertX(x, y, theta):
    x_vcs = (y - x/np.tan(theta)) * np.cos(theta) + x/np.sin(theta)
    return x_vcs


def convertY(x, y, theta):
    y_vcs = (y - x/np.tan(theta)) * np.sin(theta)
    return y_vcs


def convert2vcs(utm, ego):
    target_rel_vcs = utm
    for i in range(len(utm[0])-1):
        for j in range(len(ego[0])):
            if ego[0, j] == utm[0, i]:
                theta = -ego[4, j]
                x_utm = utm[1, j]
                y_utm = utm[2, j]
                x_vel = utm[6, j]
                y_vel = utm[7, j]
                x_a = utm[9, j]
                y_a = utm[10, j]
                x_vcs = convertX(x_utm, y_utm, theta)
                y_vcs = convertY(x_utm, y_utm, theta)
                x_vel_vcs = convertX(x_vel, y_vel, theta)
                y_vel_vcs = convertY(x_vel, y_vel, theta)
                x_a_vcs = convertX(x_a, y_a, theta)
                y_a_vcs = convertY(x_a, y_a, theta)
                target_rel_vcs[1, j] = x_vcs
                target_rel_vcs[2, j] = y_vcs
                target_rel_vcs[6, j] = x_vel_vcs
                target_rel_vcs[7, j] = y_vel_vcs
                target_rel_vcs[9, j] = x_a_vcs
                target_rel_vcs[10, j] = y_a_vcs
                break
            else:
                continue
    return target_rel_vcs


def plot(target_rel, camera):
    time = (target_rel[0, :] - target_rel[0, 0])/1000000
    time_camr = (camera[0, :] - camera[0, 0])/1000000

    # utm_y为前向（即camera_x）
    plt.figure(1)
    plt.xlabel("time(s)")
    plt.ylabel("dist_x(m)")
    plt.plot(time, target_rel[1, :], "r--", label="gt_dist_x", linewidth=2)
    plt.plot(time_camr-15, camera[3, :], "y--", label="camera_x", linewidth=2)
    plt.legend()

    plt.figure(2)
    plt.xlabel("time(s)")
    plt.ylabel("dist_y(m)")
    plt.plot(time, target_rel[2, :], "r--", label="gt_dist_y", linewidth=2)
    plt.plot(time_camr-15, camera[4, :], "y--", label="camera_y", linewidth=2)
    plt.legend()

    plt.figure(3)
    plt.xlabel("time(s)")
    plt.ylabel("heading(rad)")
    plt.plot(time, target_rel[4, :], "r--", label="gt_heading", linewidth=2)
    plt.plot(time_camr-14, camera[5, :], "y--", label="camera_heading", linewidth=2)
    plt.legend()

    plt.figure(4)
    plt.xlabel("time(s)")
    plt.ylabel("velocity_x(m/s)")
    plt.plot(time, target_rel[6, :], "r--", label="gt_velocity_x", linewidth=2)
    plt.plot(time_camr-15, camera[6, :], "y--", label="camera_velocity_x", linewidth=2)
    plt.legend()

    plt.figure(5)
    plt.xlabel("time(s)")
    plt.ylabel("velocity_y(m/s)")
    plt.plot(time, target_rel[7, :], "r--", label="gt_velocity_y", linewidth=2)
    plt.plot(time_camr-15, camera[7, :], "y--", label="camera_velocity_y", linewidth=2)
    plt.legend()

    plt.figure(6)
    plt.xlabel("time(s)")
    plt.ylabel("acceleration_x(m/s^2)")
    plt.plot(time, target_rel[9, :], "r--", label="gt_acceleration_x", linewidth=2)
    plt.plot(time_camr-15, camera[8, :], "y--", label="camera_acceleration_x", linewidth=2)
    plt.legend()

    plt.figure(7)
    plt.xlabel("time(s)")
    plt.ylabel("acceleration_y(m/s^2)")
    plt.plot(time, target_rel[10, :], "r--", label="gt_acceleration_y", linewidth=2)
    plt.plot(time_camr-15, camera[9, :], "y--", label="camera_acceleration_y", linewidth=2)
    plt.legend()

    plt.show()


if __name__ == '__main__':
    # read data from csv
    ego = np.loadtxt(open('ego.csv', 'rb'), delimiter=",", skiprows=0)
    target = np.loadtxt(open('target.csv', 'rb'), delimiter=",", skiprows=0)
    camera = np.loadtxt(open('camera_obj.csv', 'rb'), delimiter=",", skiprows=0)

    # find the intersection
    index = find_the_intersection(ego, target)

    # reshape ego and target
    ego_new = ego[:, index[0]:index[1]]
    target_new = target[:, index[2]:index[3]]

    # based on ego, match and adjust target
    target_match = match(ego_new, target_new)

    # calculate relative
    target_rel = calculate_rel(ego_new, target_match)

    # convert relative UTM to VCS(position, velocity, acceleration)
    target_rel_vcs = convert2vcs(target_rel, ego_new)

    # plot the result
    plot(target_rel_vcs, camera)
