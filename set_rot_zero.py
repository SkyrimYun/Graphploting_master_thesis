import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA, sqrt
import sophus as sp


def main():

    # load files
    try:
        pose = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/datasets/shapes_rotation/shapes_rotation/groundtruth.txt')
       

    except FileNotFoundError:
        print('Cannot open file!')
    except LookupError:
        print('Unkown unicode!')
    except UnicodeDecodeError:
        print('Decode error!')

    r0 = R.from_quat(pose[0, 4:8])
    for i in range(pose.shape[0]):
        r = R.from_quat(pose[i, 4:8])
        rot = r0.as_matrix().T.dot(r.as_matrix())
        # r_quat = R.from_matrix(rot).as_rotvec()
        # pose[i, 4] = r_quat[0]
        # pose[i, 5] = r_quat[1]
        # pose[i, 6] = r_quat[2]
        
        r_quat = R.from_matrix(rot).as_quat()
        pose[i, 4] = r_quat[0]
        pose[i, 5] = r_quat[1]
        pose[i, 6] = r_quat[2]
        pose[i, 7] = r_quat[3]

    np.savetxt(
        '/home/yunfan/work_spaces/master_thesis/datasets/shapes_rotation/shapes_rotation/groundtruth_zero.txt', pose, fmt='%.10f')


if __name__ == '__main__':
    main()
