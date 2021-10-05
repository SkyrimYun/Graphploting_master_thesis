import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA, sqrt
import sophus as sp
from matplotlib.ticker import MaxNLocator


def main():

    # load files
    try:
        gt = np.loadtxt('/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/dvspanotracking/desktop_dvspanotracking_ESIM_panorama/stamped_groundtruth.txt')
        gt_t = gt[:, 0]
        gt_vec = R.from_quat(gt[:,4:8]).as_rotvec()

        pano = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/dvspanotracking/desktop_dvspanotracking_ESIM_panorama/stamped_traj_estimate.txt')
        pano_t = pano[:, 0]
        pano_vec = R.from_quat(pano[:, 4:8]).as_rotvec()

        glo = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/globallyaligned/desktop_globallyaligned_ESIM_panorama/stamped_traj_estimate.txt')
        glo_t = glo[:, 0]
        glo_vec = R.from_quat(glo[:, 4:8]).as_rotvec()


    except FileNotFoundError:
        print('Cannot open file!')
    except LookupError:
        print('Unkown unicode!')
    except UnicodeDecodeError:
        print('Decode error!')

    fig, (ax_x, ax_y, ax_z) = plt.subplots(3, figsize=[6,12])
    # fig_x, ax_x = plt.subplots(figsize=[8.8, 4.8])
    # fig_y, ax_y = plt.subplots(figsize=[8.8, 4.8])
    # fig_z, ax_z = plt.subplots(figsize=[8.8, 4.8])


    # x axis
    ax_x.plot(pano_t, pano_vec[:, 0], color='red',linestyle='solid', lw=2.5, label='panoramic tracking')
    ax_x.plot(gt_t,   gt_vec[:, 0], color='green',linestyle='solid', lw=2.5, label='ground truth')
    ax_x.plot(glo_t,  glo_vec[:, 0],color='blue', linestyle='solid', lw=2.5, label='global aligment')
    # ax_x.plot(pose_dr_t, pose_dr_x,
    #           color='yellow', linestyle='solid', lw=2.5)
    #ax_x.set_xlim(45, 55)
    #ax_x.set_ylim(-0.4, 0.5)
    ax_x.set_title('x-axis', fontsize=20)
    ax_x.xaxis.set_major_locator(MaxNLocator(5))
    ax_x.yaxis.set_major_locator(MaxNLocator(3))
    ax_x.tick_params(axis='x', labelsize=10, direction='in')
    ax_x.tick_params(axis='y', labelsize=10, direction='in')
    #ax_x.set_ylabel('tilt [rad]',fontsize=25)

    # y axis
    ax_y.plot(pano_t, pano_vec[:, 1], color='red',linestyle='solid', lw=2.5)
    ax_y.plot(gt_t,   gt_vec[:, 1], color='green',linestyle='solid', lw=2.5)
    ax_y.plot(glo_t,  glo_vec[:, 1],color='blue', linestyle='solid', lw=2.5)
    # ax_y.plot(pose_dr_t, pose_dr_y,
    #           color='yellow', linestyle='solid', lw=2.5)
    #ax_y.set_xlim(45, 55)
    #ax_y.set_ylim(-0.6, 0.6)
    ax_y.set_title('y-axis', fontsize=20)
    ax_y.xaxis.set_major_locator(MaxNLocator(5))
    ax_y.yaxis.set_major_locator(MaxNLocator(3))
    ax_y.tick_params(axis='x', labelsize=10, direction='in')
    ax_y.tick_params(axis='y', labelsize=10, direction='in')
    #ax_y.set_ylabel('pan [rad]', fontsize=25)

    # z axis
    ax_z.plot(pano_t, pano_vec[:, 2], color='red',linestyle='solid', lw=2.5)
    ax_z.plot(gt_t,   gt_vec[:, 2], color='green',linestyle='solid', lw=2.5)
    ax_z.plot(glo_t,  glo_vec[:, 2], color='blue', linestyle='solid', lw=2.5)
    # ax_z.plot(pose_dr_t, pose_dr_z,
    #           color='yellow', linestyle='solid', lw=2.5)
    #ax_z.set_xlim(45,55)
    #ax_z.set_ylim(-1.7,0.7)
    ax_z.set_title('z-axis', fontsize=20)
    ax_z.xaxis.set_major_locator(MaxNLocator(5))
    ax_z.yaxis.set_major_locator(MaxNLocator(3))
    ax_z.tick_params(axis='x', labelsize=10, direction='in')
    ax_z.tick_params(axis='y', labelsize=10, direction='in')
    #ax_z.set_ylabel('roll [rad]', fontsize=25)
    ax_z.set_xlabel('time [s]', fontsize=25)

    
    # legend
    # lines, labels = ax_x.get_legend_handles_labels()
    # fig_x.legend(lines, labels, ncol=3, 
    #            loc='lower center', fontsize=15)

    fig.tight_layout()
   

    plt.show()


if __name__ == '__main__':
    main()
