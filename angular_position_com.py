import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA, sqrt
import sophus as sp


def main():

    # load files
    try:
        pose_gt = np.loadtxt(
            'data_gt/test_panorama/test_panorama2/pose_gt.txt')
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

        pose_glo = np.loadtxt(
            'data_globally/test_panorama/test_panorama2/position.txt')
        pose_glo_t = pose_glo[:, 0]
        pose_glo_x = pose_glo[:, 1]
        pose_glo_y = pose_glo[:, 2]
        pose_glo_z = pose_glo[:, 3]
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
        r_est = R.from_rotvec(pose_pano[i, 1:4]).as_matrix()
        r_gt = R.from_rotvec(pose_gt[index_gt, 1:4]).as_matrix()

        # RMSE
        so3_est = sp.SO3(r_est)
        so3_gt = sp.SO3(r_gt)
        error = LA.norm((so3_gt.inverse()*so3_est).log())
        rmse_pano += (error*error)

        # Geodesic distance
        tra = np.trace(r_est.dot(r_gt.T))
        theta = np.arccos((tra-1)/2)
        geodes_pano[i, 0] = theta

    rmse_pano /= pose_pano.shape[0]
    rmse_pano = sqrt(rmse_pano)
    print(rmse_pano)

    rmse_glo = 0.0
    geodes_glo = np.zeros((pose_glo.shape[0], 1))
    for i in range(pose_glo.shape[0]):
        # find cloest ground truth pose index
        est_t = pose_glo[i, 0]
        index_gt = np.argmin(np.abs(est_t-pose_gt[:, 0]))
        r_est = R.from_rotvec(pose_glo[i, 1:4]).as_matrix()
        r_gt = R.from_rotvec(pose_gt[index_gt, 1:4]).as_matrix()

        # RMSE
        so3_est = sp.SO3(r_est)
        so3_gt = sp.SO3(r_gt)
        error = LA.norm((so3_gt.inverse()*so3_est).log())
        rmse_glo += (error*error)

        # Geodesic distance
        tra = np.trace(r_est.dot(r_gt.T))
        theta = np.arccos((tra-1)/2)
        geodes_glo[i, 0] = theta

    rmse_glo /= pose_glo.shape[0]
    rmse_glo = sqrt(rmse_glo)
    print(rmse_glo)

    # gt pose compared with dvs_panotracking
    fig_pano, ax_pano = plt.subplots()
    ax_pano.plot(pose_pano_t, pose_pano_x, 'b-', label='x_pano')
    ax_pano.plot(pose_pano_t, pose_pano_y, 'y-', label='y_pano')
    ax_pano.plot(pose_pano_t, pose_pano_z, 'g-', label='z_pano')
    ax_pano.plot(pose_gt_t, pose_gt_x, 'b--', label='x_gt')
    ax_pano.plot(pose_gt_t, pose_gt_y, 'y--', label='y_gt')
    ax_pano.plot(pose_gt_t, pose_gt_z, 'g--', label='z_gt')
    ax_pano.set_xlabel('time/s')
    ax_pano.set_ylabel('rotation/rad')
    ax_pano.set_title('dvs_panotracking rotation comparison')
    ax_pano.legend()

    # gt pose compared with globally_alignment
    fig_aligned, ax_aligned = plt.subplots()
    ax_aligned.plot(pose_glo_t, pose_glo_x, 'b-', label='x_glo')
    ax_aligned.plot(pose_glo_t, pose_glo_y, 'y-', label='y_glo')
    ax_aligned.plot(pose_glo_t, pose_glo_z, 'g-', label='z_glo')
    ax_aligned.plot(pose_gt_t, pose_gt_x, 'b--', label='x_gt')
    ax_aligned.plot(pose_gt_t, pose_gt_y, 'y--', label='y_gt')
    ax_aligned.plot(pose_gt_t, pose_gt_z, 'g--', label='z_gt')
    ax_aligned.set_xlabel('time/s')
    ax_aligned.set_ylabel('rotation/rad')
    ax_aligned.set_title('Globally_Aligned_Events rotation comparison')
    ax_aligned.legend()

    # Geodesic distance between estimated rotation from globally-aligned to GT
    fig_geo_aligned, ax_geo_aligned = plt.subplots()
    ax_geo_aligned.plot(pose_glo_t, geodes_glo, 'b-')
    ax_geo_aligned.set_xlabel('time [s]')
    ax_geo_aligned.set_ylabel('Orientation error [rad]')
    ax_geo_aligned.set_title(
        'Geodesic distance between estimated rotation from Globally_Aligned_Events to GT')
    ax_geo_aligned.legend()

    # Geodesic distance between estimated rotation from panotracking to GT
    fig_geo_pano, ax_geo_pano = plt.subplots()
    ax_geo_pano.plot(pose_pano_t, geodes_pano, 'b-')
    ax_geo_pano.set_xlabel('time [s]')
    ax_geo_pano.set_ylabel('Orientation error [rad]')
    ax_geo_pano.set_title(
        'Geodesic distance between estimated rotation from Panotracking to GT')
    ax_geo_pano.legend()

    plt.show()


if __name__ == '__main__':
    main()
