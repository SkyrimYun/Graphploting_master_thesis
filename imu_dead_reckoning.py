import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import eye
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA, pi


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def main():

    # load files
    try:
        pose_gt = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/dvspanotracking/desktop_dvspanotracking_boxes_rotation/stamped_groundtruth.txt')
        pose_gt_t = pose_gt[:, 0]
        pose_gt_vec = R.from_quat(pose_gt[:, 4:8]).as_rotvec()

        pose_pano = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/dvspanotracking/desktop_dvspanotracking_boxes_rotation/stamped_traj_estimate.txt')
        pose_pano_t = pose_pano[:, 0]
        pose_pano_vec = R.from_quat(pose_pano[:, 4:8]).as_rotvec()

        pose_glo = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/globallyaligned/desktop_globallyaligned_boxes_rotation/stamped_traj_estimate.txt')
        pose_glo_t = pose_glo[:, 0]
        pose_glo_vec = R.from_quat(pose_glo[:, 4:8]).as_rotvec()

        vel_imu = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/datasets/boxes_rotation/boxes_rotation/imu.txt')
    except FileNotFoundError:
        print('Cannot open file!')
    except LookupError:
        print('Unkown unicode!')
    except UnicodeDecodeError:
        print('Decode error!')

    # try to calculate dead reckoning path from imu angular velocity
    pose_dr = np.zeros((vel_imu.shape[0], 4))
    pose_dr[:, 0] = vel_imu[:, 0]
    pose_dr[0, 1:4] = pose_gt_vec[0, :]
    for i in range(1, pose_dr.shape[0]):

        delta_t = (pose_dr[i, 0]-pose_dr[i-1, 0])
        start = R.from_rotvec(pose_dr[i-1, 1:4])
        w = vel_imu[i-1, 4:7]

        rotvec_w = w*delta_t
        rotvec_s = start.inv().as_matrix().dot(rotvec_w)
        end = start * R.from_rotvec(rotvec_s)

        pose_dr[i, 1:4] = end.as_rotvec()

    pose_dr_t = pose_dr[:, 0]
    pose_dr_x = pose_dr[:, 1]
    pose_dr_y = pose_dr[:, 2]
    pose_dr_z = pose_dr[:, 3]

    # gt imu dead reckoning pose compared with gt pose
    fig_imu, ax_imu = plt.subplots()
    ax_imu.plot(pose_dr_t, pose_dr_x, color='black',
                linestyle='solid', label='Dead Reckoning', lw=2.5)
    #ax_imu.plot(pose_dr_t, pose_dr_y, 'y-', label='y_dr')
    #ax_imu.plot(pose_dr_t, pose_dr_z, 'g-', label='z_dr')
    ax_imu.plot(pose_gt_t, pose_gt_vec[:, 0],
                'g--', label='GroundTruth', lw=2.5)
    #ax_imu.plot(pose_gt_t, pose_gt_vec[:,1], 'y--', label='y_gt')
    #ax_imu.plot(pose_gt_t, pose_gt_vec[:,2], 'g--', label='z_gt')
    ax_imu.plot(pose_pano_t, pose_pano_vec[:, 0],
                'r-.', label='Panoramic Tracking', lw=2.5)
    #ax_imu.plot(pose_pano_t, pose_pano_vec[:,1], 'y-.', label='y_pano')
    #ax_imu.plot(pose_pano_t, pose_pano_vec[:,2], 'g-.', label='z_pano')
    ax_imu.plot(pose_glo_t, pose_glo_vec[:, 0],
                'b-.', label='Global Alignment', lw=2.5)

    ax_imu.set_xlim(0, 15)
    ax_imu.set_ylim(-1, 1)
    ax_imu.set_xlabel('time [s]', fontsize=15)
    ax_imu.set_ylabel('rotation [rad]', fontsize=15)
    #ax_imu.set_title('IMU Dead Reckoning rotation comparison')
    ax_imu.legend()
    fig_imu.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
