import pandas as pd
import os
import numpy as np

def traverse(filepath):
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)
        file = os.path.join(filepath, fi_d)
        if os.path.getsize(file) == 0:
            os.remove(file)

def get_file(path):
    files = os.listdir(path)
    files.sort()
    list = []
    for file in files:
        # filename = str(file)
        list.append(file)
    # return (list)
    return list

if __name__ == '__main__':
    # path = '/media/yong/CAMR_1/FAW_DATA/GroundTruth/20210531/DATA/ego_txt/'
    path = '/media/yong/CAMR_1/FAW_DATA/GroundTruth/20210531/DATA/target_txt/'
    # path = '/media/yong/CAMR_1/FAW_DATA/GroundTruth/20210531/DATA/camera_obj/'
    traverse(path)
    list = get_file(path)
    pds = []
    for i in range(len(list)):
        df = pd.read_csv(path + list[i], sep=' ', header=None)
        pds.append(df)

    df_ = pd.concat(pds, axis=1)
    # df_.to_csv('ego.csv', mode='a', header=None, index=False)
    df_.to_csv('target.csv', mode='a', header=None, index=False)
    # df_.to_csv('camera_obj.csv', mode='a', header=None, index=False)