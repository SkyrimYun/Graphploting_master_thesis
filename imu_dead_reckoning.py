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
        vel_gt = np.loadtxt('data_gt/test_panorama/imu_angular.txt')
        vel_gt_t = vel_gt[:, 0]
        vel_gt_x = vel_gt[:, 1]
        vel_gt_y = vel_gt[:, 2]
        vel_gt_z = vel_gt[:, 3]

        pose_gt = np.loadtxt('data_gt/test_panorama/pose_gt.txt')
        pose_gt_t = pose_gt[:, 0]
        pose_gt_x = pose_gt[:, 1]
        pose_gt_y = pose_gt[:, 2]
        pose_gt_z = pose_gt[:, 3]

        pose_pano = np.loadtxt('data_panotracking/esim/estimated_pose.txt')
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
    pose_dr = vel_gt.copy()
    pose_dr[0, 1:4] = pose_gt[10, 1:4]
    for i in range(1, pose_dr.shape[0]):
        # rotvec
        delta_t = (pose_dr[i, 0]-pose_dr[i-1, 0])
        r_vec1 = pose_dr[i-1, 1:4]
        w = vel_gt[i-1, 1:4]
        r_vec2 = w*delta_t+r_vec1
        pose_dr[i, 1:4] = r_vec2

        # matrix
        # r_vec1 = R.from_rotvec(pose_dr[i-1, 1:4])
        # R_vec1 = r_vec1.as_matrix()
        # w = vel_gt[i-1, 1:4]
        # s_w = skew(w)
        # delta_t = (pose_dr[i, 0]-pose_dr[i-1, 0])
        # delta_R = (s_w*delta_t+np.eye(3))
        # R_vec2 = delta_R*R_vec1
        # pose_dr[i, 1:4] = R.from_matrix(R_vec2).as_rotvec()

        # quat
        # r_vec1 = R.from_rotvec(pose_dr[i-1, 1:4])
        # w = vel_gt[i-1, 1:4]
        # delta_t = (pose_dr[i, 0]-pose_dr[i-1, 0])
        # qw = R.from_quat([0, w[0], w[1], w[2]])
        # pDot = r_vec1*qw
        # q2 = r_vec1.as_quat()+0.5*pDot.as_quat()*delta_t
        # q2 = q2 / LA.norm(q2)
        # pose_dr[i, 1:4] = R.from_quat(q2).as_rotvec()

    pose_dr_t = pose_dr[:, 0]
    pose_dr_x = -pose_dr[:, 1]
    pose_dr_y = -pose_dr[:, 3]
    pose_dr_z = -pose_dr[:, 2]

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
