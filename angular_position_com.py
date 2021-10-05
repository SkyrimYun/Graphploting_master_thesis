import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA, sqrt
import sophus as sp


def main():

    # load files
    try:
        pose_gt = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/globallyaligned/desktop_globallyaligned_fast_motion/stamped_groundtruth.txt')
        pose_gt_t = pose_gt[:, 0]
        pose_gt_vec = R.from_quat(pose_gt[:, 4:8]).as_rotvec()

        pose_gt_x = pose_gt[:, 4]
        pose_gt_y = pose_gt[:, 5]
        pose_gt_z = pose_gt[:, 6]
        pose_gt_w = pose_gt[:, 7]

        pose_pano = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/dvspanotracking/desktop_dvspanotracking_fast_motion/stamped_traj_estimate.txt')
        pose_pano_t = pose_pano[:, 0]
        pose_pano_vec = R.from_quat(pose_pano[:, 4:8]).as_rotvec()

        pose_pano_x = pose_pano[:, 4]
        pose_pano_y = pose_pano[:, 5]
        pose_pano_z = pose_pano[:, 6]
        pose_pano_w = pose_pano[:, 7]

        pose_glo = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/dvs_smt/src/dvs_smt/data/fast_motion/position_rpg.txt')
        pose_glo_t = pose_glo[:, 0]
        pose_glo_vec = R.from_quat(pose_glo[:, 4:8]).as_rotvec()

        pose_glo_x = pose_glo[:, 4]
        pose_glo_y = pose_glo[:, 5]
        pose_glo_z = pose_glo[:, 6]
        pose_glo_w = pose_glo[:, 7]

    except FileNotFoundError:
        print('Cannot open file!')
    except LookupError:
        print('Unkown unicode!')
    except UnicodeDecodeError:
        print('Decode error!')

    # calculate error
    rmse_pano = 0.0
    geodes_pano = np.zeros((pose_pano.shape[0], 1))
    for i in range(pose_pano.shape[0]):
        # find cloest ground truth pose index
        est_t = pose_pano[i, 0]
        index_gt = np.argmin(np.abs(est_t-pose_gt[:, 0]))
        r_est = R.from_quat(pose_pano[i, 4:8]).as_matrix()
        r_gt = R.from_quat(pose_gt[index_gt, 4:8]).as_matrix()

        # RMSE
        so3_est = sp.SO3(r_est)
        so3_gt = sp.SO3(r_gt)
        error = LA.norm((so3_gt.inverse()*so3_est).log())
        rmse_pano += (error*error)

        # Geodesic distance
        tra = np.trace(r_est.dot(r_gt.T))
        theta = np.arccos((tra-1)/2)
        geodes_pano[i, 0] = theta*180/np.pi

    rmse_pano /= pose_pano.shape[0]
    rmse_pano = sqrt(rmse_pano)
    print(rmse_pano*180/np.pi)

    rmse_glo = 0.0
    rmse_glo_x = 0.0
    rmse_glo_y = 0.0
    rmse_glo_z = 0.0
    geodes_glo = np.zeros((pose_glo.shape[0], 1))
    for i in range(pose_glo.shape[0]):
        # find cloest ground truth pose index
        est_t = pose_glo[i, 0]
        index_gt = np.argmin(np.abs(est_t-pose_gt[:, 0]))
        R_est = R.from_quat(pose_glo[i, 4:8]).as_matrix()
        R_gt = R.from_quat(pose_gt[index_gt, 4:8]).as_matrix()

        # RMSE
        so3_est = sp.SO3(R_est)
        so3_gt = sp.SO3(R_gt)
        error = LA.norm((so3_gt.inverse()*so3_est).log())
        rmse_glo += (error*error)

        # RMSE Kim's paper
        r_est = R.from_quat(pose_glo[i, 4:8]).as_rotvec()
        r_gt = R.from_quat(pose_gt[index_gt, 4:8]).as_rotvec()
        error_x = (r_est[0]-r_gt[0]) ** 2
        error_y = (r_est[1]-r_gt[1]) ** 2
        error_z = (r_est[2]-r_gt[2]) ** 2
        rmse_glo_x += error_x
        rmse_glo_y += error_y
        rmse_glo_z += error_z

        # Geodesic distance
        tra = np.trace(R_est.dot(R_gt.T))
        theta = np.arccos((tra-1)/2)
        geodes_glo[i, 0] = theta*180/np.pi

    rmse_glo /= pose_glo.shape[0]
    rmse_glo = sqrt(rmse_glo)
    print(rmse_glo*180/np.pi)
    rmse_glo_x /= pose_glo.shape[0]
    rmse_glo_y /= pose_glo.shape[0]
    rmse_glo_z /= pose_glo.shape[0]
    rmse_glo_x = sqrt(rmse_glo_x)
    rmse_glo_y = sqrt(rmse_glo_y)
    rmse_glo_z = sqrt(rmse_glo_z)
    print(rmse_glo_x * 180/np.pi)
    print(rmse_glo_y * 180/np.pi)
    print(rmse_glo_z * 180/np.pi)

    # gt pose compared with dvs_panotracking
    fig_pano, ax_pano = plt.subplots()
    ax_pano.plot(pose_pano_t, pose_pano_vec[:, 0], 'b-', label='x_est')
    ax_pano.plot(pose_pano_t, pose_pano_vec[:, 1], 'y-', label='y_est')
    ax_pano.plot(pose_pano_t, pose_pano_vec[:, 2], 'g-', label='z_est')
    #ax_pano.plot(pose_pano_t, pose_pano_w, 'r-', label='w_pano')

    ax_pano.plot(pose_gt_t, pose_gt_vec[:, 0], 'b--', label='x_gt')
    ax_pano.plot(pose_gt_t, pose_gt_vec[:, 1], 'y--', label='y_gt')
    ax_pano.plot(pose_gt_t, pose_gt_vec[:, 2], 'g--', label='z_gt')
    #ax_pano.plot(pose_gt_t, pose_gt_w, 'r--', label='w_gt')

    ax_pano.set_xlabel('time [s]')
    ax_pano.set_ylabel('rotation [rad]')
    ax_pano.set_title('dvs_panotracking rotation comparison')
    ax_pano.legend()

    # gt pose compared with globally_alignment
    fig_aligned, ax_aligned = plt.subplots()
    ax_aligned.plot(
        pose_glo_t, pose_glo_vec[:, 0] * 180/np.pi, 'b-', label='x_est')
    ax_aligned.plot(
        pose_glo_t, pose_glo_vec[:, 1] * 180/np.pi, 'y-', label='y_est')
    ax_aligned.plot(
        pose_glo_t, pose_glo_vec[:, 2] * 180/np.pi, 'g-', label='z_est')
    #ax_aligned.plot(pose_glo_t, pose_glo_w, 'r-', label='w_glo')

    ax_aligned.plot(
        pose_gt_t, pose_gt_vec[:, 0] * 180/np.pi, 'b--', label='x_gt')
    ax_aligned.plot(
        pose_gt_t, pose_gt_vec[:, 1] * 180/np.pi, 'y--', label='y_gt')
    ax_aligned.plot(
        pose_gt_t, pose_gt_vec[:, 2] * 180/np.pi, 'g--', label='z_gt')
    #ax_aligned.plot(pose_gt_t, pose_gt_w, 'r--', label='w_gt')

    ax_aligned.set_xlabel('time [s]')
    ax_aligned.set_ylabel('rotation [deg]')
    ax_aligned.set_title('Globally_Aligned_Events rotation comparison')
    ax_aligned.legend()

    # Geodesic distance between estimated rotation from globally-aligned to GT
    fig_geo_aligned, ax_geo_aligned = plt.subplots()
    ax_geo_aligned.plot(pose_glo_t, geodes_glo, 'b-')
    ax_geo_aligned.set_xlabel('time [s]')
    ax_geo_aligned.set_ylabel('Orientation error [deg]')
    ax_geo_aligned.set_title(
        'Geodesic distance between estimated rotation from Globally_Aligned_Events to GT')

    # Geodesic distance between estimated rotation from panotracking to GT
    fig_geo_pano, ax_geo_pano = plt.subplots()
    ax_geo_pano.plot(pose_pano_t, geodes_pano, 'b-')
    ax_geo_pano.set_xlabel('time [s]')
    ax_geo_pano.set_ylabel('Orientation error [deg]')
    ax_geo_pano.set_title(
        'Geodesic distance between estimated rotation from Panotracking to GT')

    plt.show()


if __name__ == '__main__':
    main()
