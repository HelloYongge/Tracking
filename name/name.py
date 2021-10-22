import os
from shutil import copyfile

path = '/media/yong/CAMR_1/smoke_test_case/10_FrSh-MultiFramMultiObj-Track_2020-09-03_15-59-07/FrSh-MultiFramMultiObj-Track'
files = os.listdir(path)

for fi in files:
    name = fi.split('.')[0]
    print(name)
    copyfile("/media/yong/CAMR_1/smoke_test_case/10_FrSh-MultiFramMultiObj-Track_2020-09-03_15-59-07/FrSh-MultiFramMultiObj-Track/1599119948340650_camr_sensor_obj", name + '_camr_sensor_obj')
