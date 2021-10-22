import csv
import matplotlib.pyplot as plt
import numpy as np

def import_data():
    pass

def transform():
    box_x = 630.254
    box_y = 497.362
    box_w = 13.4822
    box_h = 18.7445
    box_H = 3.3
    box_trans_x = box_x + box_w/2
    box_trans_y = box_y + box_h
    f_y = 1426.34
    Hc = 2.4115
    vp = 494.725
    intrinsic_mat_inv = np.mat([[0.00070108, 0, -0.460102],
                                [0, 0.000701093, -0.349554],
                                [0, 0, 1]])
    camera2baselink = np.mat([[-0.0085462, -0.00270595, 0.99996, 6.31329],
                              [-0.99974, -0.0211253, -0.00860149, 0.0605919],
                              [0.0211478, -0.999773, -0.0025247, 2.4115],
                              [0, 0, 0, 1]])
    d1 = f_y * box_H/box_h
    y = box_y + box_h
    d2 = f_y * Hc/(y - vp)
    d = (d1 + d2)/2
    pt2d_ex = np.mat([box_trans_x, box_trans_y, 1.0]).T
    pt3d_cam = intrinsic_mat_inv * d * pt2d_ex
    pt3d_cam_ex = np.mat([pt3d_cam[0], pt3d_cam[1], pt3d_cam[2], 1.0]).T
    pt3d_ego = camera2baselink * pt3d_cam_ex
    return pt3d_ego

def kalmanfilter(pt):
    pass

if __name__ == '__main__':
    # import_data()
    pt3d_transform = transform()
    print(pt3d_transform)
    # kalmanfilter(pt3d_transform)
