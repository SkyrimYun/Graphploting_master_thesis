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
            '/home/yunfan/work_spaces/master_thesis/datasets/2021-06-04-11-21-58/imu_angular.txt')

        pose_glo = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/Globally_Aligned_Events/src/rotation_estimator/data/real_world/2021-06-04-11-21-58/position_rpg.txt')
        pose_glo_t = pose_glo[:, 0]
        pose_glo_x = pose_glo[:, 4]
        pose_glo_y = pose_glo[:, 5]
        pose_glo_z = pose_glo[:, 6]
        pose_glo_w = pose_glo[:, 7]

        pose_pano = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/dvs-panotracking/data/real_world/output_pose/estimated_pose_rpg.txt')
        pose_pano_t = pose_pano[:, 0]
        pose_pano_x = pose_pano[:, 4]
        pose_pano_y = pose_pano[:, 5]
        pose_pano_z = pose_pano[:, 6]
        pose_pano_w = pose_pano[:, 7]

    except FileNotFoundError:
        print('Cannot open file!')
    except LookupError:
        print('Unkown unicode!')
    except UnicodeDecodeError:
        print('Decode error!')

    # try to calculate dead reckoning path from imu angular velocity
    pose_dr = np.zeros((vel_imu.shape[0],5))
    pose_dr[:,0]=vel_imu[:,0]
    pose_dr[0, 1:5] = np.array([0,0,0,1])
    for i in range(1, pose_dr.shape[0]):

        delta_t = (pose_dr[i, 0]-pose_dr[i-1, 0])
        start = R.from_quat(pose_dr[i-1, 1:5])
        w = vel_imu[i-1, 1:4]

        rotvec_w = w*delta_t
        rotvec_s = start.inv().as_matrix().dot(rotvec_w)
        end = start * R.from_rotvec(rotvec_s)

        pose_dr[i, 1:5] = end.as_quat()

    pose_dr_t = pose_dr[:, 0]
    pose_dr_x = pose_dr[:, 1]
    pose_dr_y = pose_dr[:, 2]
    pose_dr_z = pose_dr[:, 3]
    pose_dr_w = pose_dr[:, 4]


    #imu dead reckoning pose compared with glo pose
    fig_imu, ax_imu = plt.subplots()
    ax_imu.plot(pose_dr_t, pose_dr_x, 'b-', label='x_dr')
    ax_imu.plot(pose_dr_t, pose_dr_y, 'y-', label='y_dr')
    ax_imu.plot(pose_dr_t, pose_dr_z, 'g-', label='z_dr')
    ax_imu.plot(pose_dr_t, pose_dr_w, 'r-', label='w_dr')

    ax_imu.plot(pose_glo_t, pose_glo_x, 'b--', label='x_glo')
    ax_imu.plot(pose_glo_t, pose_glo_y, 'y--', label='y_glo')
    ax_imu.plot(pose_glo_t, pose_glo_z, 'g--', label='z_glo')
    ax_imu.plot(pose_glo_t, pose_glo_w, 'r--', label='w_glo')
    ax_imu.set_xlabel('time [s]')
    ax_imu.set_ylabel('rotation [rad]')
    ax_imu.set_title('IMU Dead Reckoning rotation comparison')
    ax_imu.legend()

    #imu dead reckoning pose compared with pano pose
    fig_imu, ax_imu = plt.subplots()
    ax_imu.plot(pose_dr_t, pose_dr_x, 'b-', label='x_dr')
    ax_imu.plot(pose_dr_t, pose_dr_y, 'y-', label='y_dr')
    ax_imu.plot(pose_dr_t, pose_dr_z, 'g-', label='z_dr')
    ax_imu.plot(pose_dr_t, pose_dr_w, 'r-', label='w_dr')

    ax_imu.plot(pose_pano_t, pose_pano_x, 'b--', label='x_pano')
    ax_imu.plot(pose_pano_t, pose_pano_y, 'y--', label='y_pano')
    ax_imu.plot(pose_pano_t, pose_pano_z, 'g--', label='z_pano')
    ax_imu.plot(pose_pano_t, pose_pano_w, 'r--', label='w_pano')
    ax_imu.set_xlabel('time [s]')
    ax_imu.set_ylabel('rotation [rad]')
    ax_imu.set_title('IMU Dead Reckoning rotation comparison')
    ax_imu.legend()

    #glo pose reckoning pose compared with pano pose
    fig_imu, ax_imu = plt.subplots()
    ax_imu.plot(pose_glo_t, pose_glo_x, 'b-', label='x_glo')
    ax_imu.plot(pose_glo_t, pose_glo_y, 'y-', label='y_glo')
    ax_imu.plot(pose_glo_t, pose_glo_z, 'g-', label='z_glo')
    ax_imu.plot(pose_glo_t, pose_glo_w, 'r-', label='w_glo')

    ax_imu.plot(pose_pano_t, pose_pano_x, 'b--', label='x_pano')
    ax_imu.plot(pose_pano_t, pose_pano_y, 'y--', label='y_pano')
    ax_imu.plot(pose_pano_t, pose_pano_z, 'g--', label='z_pano')
    ax_imu.plot(pose_pano_t, pose_pano_w, 'r--', label='w_pano')
    ax_imu.set_xlabel('time [s]')
    ax_imu.set_ylabel('rotation [rad]')
    ax_imu.set_title('dvs-panotracking and Globally-alined rotation comparison')
    ax_imu.legend()

    plt.show()


if __name__ == '__main__':
    main()
