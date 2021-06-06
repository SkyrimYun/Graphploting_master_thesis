import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import eye
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def main():

    # load files
    try:
        vel_imu = np.loadtxt(
            'data_gt/test_panorama/test_panorama3/imu_angular.txt')
        vel_imu_t = vel_imu[:, 0]
        vel_imu_x = vel_imu[:, 1]
        vel_imu_y = vel_imu[:, 2]
        vel_imu_z = vel_imu[:, 3]

        pose_gt = np.loadtxt(
            'data_gt/test_panorama/test_panorama3/pose_gt.txt')
        pose_gt_t = pose_gt[:, 0]
        pose_gt_x = pose_gt[:, 1]
        pose_gt_y = pose_gt[:, 2]
        pose_gt_z = pose_gt[:, 3]

        pose_pano = np.loadtxt(
            'data_panotracking/esim/test_panorama2/estimated_pose.txt')
        pose_pano_t = pose_pano[:, 0]
        pose_pano_x = pose_pano[:, 1]
        pose_pano_y = pose_pano[:, 2]
        pose_pano_z = pose_pano[:, 3]
    except FileNotFoundError:
        print('Cannot open file!')
    except LookupError:
        print('Unkown unicode!')
    except UnicodeDecodeError:
        print('Decode error!')

    # try to calculate dead reckoning path from imu angular velocity
    pose_dr = vel_imu.copy()
    pose_dr[0, 1:4] = pose_gt[0, 1:4]
    for i in range(1, pose_dr.shape[0]):

        delta_t = (pose_dr[i, 0]-pose_dr[i-1, 0])
        start = R.from_rotvec(pose_dr[i-1, 1:4])
        w = vel_imu[i-1, 1:4]

        rotvec_w = w*delta_t
        rotvec_s = start.inv().as_matrix().dot(rotvec_w)
        end = start * R.from_rotvec(rotvec_s)

        pose_dr[i, 1:4] = end.as_rotvec()

    pose_dr_t = pose_dr[:, 0]
    pose_dr_x = -pose_dr[:, 1]
    pose_dr_y = -pose_dr[:, 2]
    pose_dr_z = -pose_dr[:, 3]

    # gt imu dead reckoning pose compared with gt pose
    fig_imu, ax_imu = plt.subplots()
    ax_imu.plot(pose_dr_t, pose_dr_x, 'b-', label='x_dr')
    ax_imu.plot(pose_dr_t, pose_dr_y, 'y-', label='y_dr')
    ax_imu.plot(pose_dr_t, pose_dr_z, 'g-', label='z_dr')
    ax_imu.plot(pose_gt_t, pose_gt_x, 'b--', label='x_gt')
    ax_imu.plot(pose_gt_t, pose_gt_y, 'y--', label='y_gt')
    ax_imu.plot(pose_gt_t, pose_gt_z, 'g--', label='z_gt')
    ax_imu.plot(pose_pano_t, pose_pano_x, 'b-.', label='x_pano')
    ax_imu.plot(pose_pano_t, pose_pano_y, 'y-.', label='y_pano')
    ax_imu.plot(pose_pano_t, pose_pano_z, 'g-.', label='z_pano')
    ax_imu.set_xlabel('time/s')
    ax_imu.set_ylabel('rotation/rad')
    ax_imu.set_title('IMU Dead Reckoning rotation comparison')
    ax_imu.legend()

    plt.show()


if __name__ == '__main__':
    main()
