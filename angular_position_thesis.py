import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA, sqrt
import sophus as sp
from matplotlib.ticker import MaxNLocator


def main():

    # load files
    try:
        boxes_gt = np.loadtxt('/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/dvspanotracking/desktop_dvspanotracking_boxes_rotation/stamped_groundtruth.txt')
        boxes_gt_t = boxes_gt[:, 0]
        boxes_gt_vec = R.from_quat(boxes_gt[:,4:8]).as_rotvec()
        
        boxes_pano = np.loadtxt('/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/dvspanotracking/desktop_dvspanotracking_boxes_rotation/stamped_traj_estimate.txt')
        boxes_pano_t = boxes_pano[:, 0]
        boxes_pano_vec = R.from_quat(boxes_pano[:, 4:8]).as_rotvec()

        boxes_glo = np.loadtxt('/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/globallyaligned/desktop_globallyaligned_boxes_rotation/stamped_traj_estimate.txt')
        boxes_glo_t = boxes_glo[:, 0]
        boxes_glo_vec = R.from_quat(boxes_glo[:, 4:8]).as_rotvec()

        dynamic_gt = np.loadtxt('/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/dvspanotracking/desktop_dvspanotracking_dynamic_rotation/stamped_groundtruth.txt')
        dynamic_gt_t = dynamic_gt[:, 0]
        dynamic_gt_vec = R.from_quat(dynamic_gt[:, 4:8]).as_rotvec()

        dynamic_pano = np.loadtxt('/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/dvspanotracking/desktop_dvspanotracking_dynamic_rotation/stamped_traj_estimate.txt')
        dynamic_pano_t = dynamic_pano[:, 0]
        dynamic_pano_vec = R.from_quat(dynamic_pano[:, 4:8]).as_rotvec()

        dynamic_glo = np.loadtxt('/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/globallyaligned/desktop_globallyaligned_dynamic_rotation/stamped_traj_estimate.txt')
        dynamic_glo_t = dynamic_glo[:, 0]
        dynamic_glo_vec = R.from_quat(dynamic_glo[:, 4:8]).as_rotvec()

        poster_gt = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/dvspanotracking/desktop_dvspanotracking_poster_rotation/stamped_groundtruth.txt')
        poster_gt_t = poster_gt[:, 0]
        poster_gt_vec = R.from_quat(poster_gt[:, 4:8]).as_rotvec()

        poster_pano = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/dvspanotracking/desktop_dvspanotracking_poster_rotation/stamped_traj_estimate.txt')
        poster_pano_t = poster_pano[:, 0]
        poster_pano_vec = R.from_quat(poster_pano[:, 4:8]).as_rotvec()

        poster_glo = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/globallyaligned/desktop_globallyaligned_poster_rotation/stamped_traj_estimate.txt')
        poster_glo_t = poster_glo[:, 0]
        poster_glo_vec = R.from_quat(poster_glo[:, 4:8]).as_rotvec()

        panorama_gt = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/dvspanotracking/desktop_dvspanotracking_ESIM_panorama/stamped_groundtruth.txt')
        panorama_gt_t = panorama_gt[:, 0]
        panorama_gt_vec = R.from_quat(panorama_gt[:, 4:8]).as_rotvec()

        panorama_pano = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/dvspanotracking/desktop_dvspanotracking_ESIM_panorama/stamped_traj_estimate.txt')
        panorama_pano_t = panorama_pano[:, 0]
        panorama_pano_vec = R.from_quat(panorama_pano[:, 4:8]).as_rotvec()

        panorama_glo = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/rpg_trajectory_evaluation/results/cmp_tracking/desktop/globallyaligned/desktop_globallyaligned_ESIM_panorama/stamped_traj_estimate.txt')
        panorama_glo_t = panorama_glo[:, 0]
        panorama_glo_vec = R.from_quat(panorama_glo[:, 4:8]).as_rotvec()

        vel_imu = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/datasets/boxes_rotation/boxes_rotation/imu.txt')

    except FileNotFoundError:
        print('Cannot open file!')
    except LookupError:
        print('Unkown unicode!')
    except UnicodeDecodeError:
        print('Decode error!')

    # calculate dead reckoning path from imu angular velocity
    # pose_dr = np.zeros((vel_imu.shape[0],4))
    # pose_dr[:,0] = vel_imu[:,0]
    # pose_dr[0, 1:4] = pose_gt_vec[0, :]
    # for i in range(1, pose_dr.shape[0]):

    #     delta_t = (pose_dr[i, 0]-pose_dr[i-1, 0])
    #     start = R.from_rotvec(pose_dr[i-1, 1:4])
    #     w = vel_imu[i-1, 4:7]

    #     rotvec_w = w*delta_t
    #     rotvec_s = start.inv().as_matrix().dot(rotvec_w)
    #     end = start * R.from_rotvec(rotvec_s)

    #     pose_dr[i, 1:4] = end.as_rotvec()

    # pose_dr_t = pose_dr[:, 0]
    # pose_dr_x = pose_dr[:, 1]
    # pose_dr_y = pose_dr[:, 2]
    # pose_dr_z = pose_dr[:, 3]

    fig, ((ax_boxes_x, ax_dynamic_x, ax_poster_x, ax_panorama_x), 
          (ax_boxes_y, ax_dynamic_y, ax_poster_y, ax_panorama_y), 
          (ax_boxes_z, ax_dynamic_z, ax_poster_z, ax_panorama_z)) = plt.subplots(3, 4)
    #fig_boxes.suptitle('boxes_rotation', fontsize=25)
    #fig_dyanmic.suptitle('dynamic_rotation', fontsize=25)


    # boxes rotation
    # x axis
    ax_boxes_x.plot(boxes_pano_t, boxes_pano_vec[:, 0], color='red',linestyle='solid', lw=2.5, label='panoramic tracking')
    ax_boxes_x.plot(boxes_gt_t, boxes_gt_vec[:, 0], color='black',linestyle='dashed', lw=2.5, label='ground truth')
    ax_boxes_x.plot(boxes_glo_t, boxes_glo_vec[:, 0],color='blue', linestyle='solid', lw=2.5, label='global aligment')
    # ax_x.plot(pose_dr_t, pose_dr_x,
    #           color='yellow', linestyle='solid', lw=2.5)
    ax_boxes_x.set_xlim(45, 55)
    ax_boxes_x.set_ylim(-0.9, 0.6)
    ax_boxes_x.set_title('boxes_rotation \n x-axis', fontsize=20)
    ax_boxes_x.xaxis.set_major_locator(MaxNLocator(5))
    ax_boxes_x.yaxis.set_major_locator(MaxNLocator(3))
    ax_boxes_x.tick_params(axis='x', labelsize=10, direction='in')
    ax_boxes_x.tick_params(axis='y', labelsize=10, direction='in')
    ax_boxes_x.set_ylabel('tilt [rad]',fontsize=15)

    # y axis
    ax_boxes_y.plot(boxes_pano_t, boxes_pano_vec[:, 1], color='red',linestyle='solid', lw=2.5)
    ax_boxes_y.plot(boxes_gt_t, boxes_gt_vec[:, 1], color='black',linestyle='dashed', lw=2.5)
    ax_boxes_y.plot(boxes_glo_t, boxes_glo_vec[:, 1],color='blue', linestyle='solid', lw=2.5)
    # ax_y.plot(pose_dr_t, pose_dr_y,
    #           color='yellow', linestyle='solid', lw=2.5)
    ax_boxes_y.set_xlim(45, 55)
    ax_boxes_y.set_ylim(-0.8, 0.8)
    ax_boxes_y.set_title('y-axis', fontsize=20)
    ax_boxes_y.xaxis.set_major_locator(MaxNLocator(5))
    ax_boxes_y.yaxis.set_major_locator(MaxNLocator(3))
    ax_boxes_y.tick_params(axis='x', labelsize=10, direction='in')
    ax_boxes_y.tick_params(axis='y', labelsize=10, direction='in')
    ax_boxes_y.set_ylabel('pan [rad]', fontsize=15)

    # z axis
    ax_boxes_z.plot(boxes_pano_t, boxes_pano_vec[:, 2], color='red',linestyle='solid', lw=2.5)
    ax_boxes_z.plot(boxes_gt_t, boxes_gt_vec[:, 2], color='black',linestyle='dashed', lw=2.5)
    ax_boxes_z.plot(boxes_glo_t, boxes_glo_vec[:, 2], color='blue', linestyle='solid', lw=2.5)
    # ax_z.plot(pose_dr_t, pose_dr_z,
    #           color='yellow', linestyle='solid', lw=2.5)
    ax_boxes_z.set_xlim(45,55)
    ax_boxes_z.set_ylim(-1.3,1.2)
    ax_boxes_z.set_title('z-axis', fontsize=20)
    ax_boxes_z.xaxis.set_major_locator(MaxNLocator(5))
    ax_boxes_z.yaxis.set_major_locator(MaxNLocator(3))
    ax_boxes_z.tick_params(axis='x', labelsize=10, direction='in')
    ax_boxes_z.tick_params(axis='y', labelsize=10, direction='in')
    ax_boxes_z.set_ylabel('roll [rad]', fontsize=15)
    ax_boxes_z.set_xlabel('time [s]', fontsize=15)

    
    # dynamic rotation
    # x axis
    ax_dynamic_x.plot(
        dynamic_pano_t, dynamic_pano_vec[:, 0], color='red', linestyle='solid', lw=2.5)
    ax_dynamic_x.plot(
        dynamic_gt_t, dynamic_gt_vec[:, 0], color='black', linestyle='dashed', lw=2.5)
    ax_dynamic_x.plot(
        dynamic_glo_t, dynamic_glo_vec[:, 0], color='blue', linestyle='solid', lw=2.5)
    # ax_x.plot(pose_dr_t, pose_dr_x,
    #           color='yellow', linestyle='solid', lw=2.5)
    ax_dynamic_x.set_xlim(45, 55)
    ax_dynamic_x.set_ylim(-0.4, 0.6)
    ax_dynamic_x.set_title('dynamic_rotation \n x-axis', fontsize=20)
    ax_dynamic_x.xaxis.set_major_locator(MaxNLocator(5))
    ax_dynamic_x.yaxis.set_major_locator(MaxNLocator(3))
    ax_dynamic_x.tick_params(axis='x', labelsize=10, direction='in')
    ax_dynamic_x.tick_params(axis='y', labelsize=10, direction='in')

    # y axis
    ax_dynamic_y.plot(
        dynamic_pano_t, dynamic_pano_vec[:, 1], color='red', linestyle='solid', lw=2.5)
    ax_dynamic_y.plot(
        dynamic_gt_t, dynamic_gt_vec[:, 1], color='black', linestyle='dashed', lw=2.5)
    ax_dynamic_y.plot(
        dynamic_glo_t, dynamic_glo_vec[:, 1], color='blue', linestyle='solid', lw=2.5)
    # ax_y.plot(pose_dr_t, pose_dr_y,
    #           color='yellow', linestyle='solid', lw=2.5)
    ax_dynamic_y.set_xlim(45, 55)
    ax_dynamic_y.set_ylim(-0.6, 0.6)
    ax_dynamic_y.set_title('y-axis', fontsize=20)
    ax_dynamic_y.xaxis.set_major_locator(MaxNLocator(5))
    ax_dynamic_y.yaxis.set_major_locator(MaxNLocator(3))
    ax_dynamic_y.tick_params(axis='x', labelsize=10, direction='in')
    ax_dynamic_y.tick_params(axis='y', labelsize=10, direction='in')

    # z axis
    ax_dynamic_z.plot(
        dynamic_pano_t, dynamic_pano_vec[:, 2], color='red', linestyle='solid', lw=2.5)
    ax_dynamic_z.plot(
        dynamic_gt_t, dynamic_gt_vec[:, 2], color='black', linestyle='dashed', lw=2.5)
    ax_dynamic_z.plot(
        dynamic_glo_t, dynamic_glo_vec[:, 2], color='blue', linestyle='solid', lw=2.5)
    # ax_z.plot(pose_dr_t, pose_dr_z,
    #           color='yellow', linestyle='solid', lw=2.5)
    ax_dynamic_z.set_xlim(45, 55)
    ax_dynamic_z.set_ylim(-1.6, 0.7)
    ax_dynamic_z.set_title('z-axis', fontsize=20)
    ax_dynamic_z.xaxis.set_major_locator(MaxNLocator(5))
    ax_dynamic_z.yaxis.set_major_locator(MaxNLocator(3))
    ax_dynamic_z.tick_params(axis='x', labelsize=10, direction='in')
    ax_dynamic_z.tick_params(axis='y', labelsize=10, direction='in')
    ax_dynamic_z.set_xlabel('time [s]', fontsize=15)

 

    # poster rotation
    # x axis
    ax_poster_x.plot(
        poster_pano_t, poster_pano_vec[:, 0], color='red', linestyle='solid', lw=2.5)
    ax_poster_x.plot(
        poster_gt_t, poster_gt_vec[:, 0], color='black', linestyle='dashed', lw=2.5)
    ax_poster_x.plot(
        poster_glo_t, poster_glo_vec[:, 0], color='blue', linestyle='solid', lw=2.5)
    # ax_x.plot(pose_dr_t, pose_dr_x,
    #           color='yellow', linestyle='solid', lw=2.5)
    ax_poster_x.set_xlim(45, 55)
    ax_poster_x.set_ylim(-0.7, 0.5)
    ax_poster_x.set_title('poster_rotation \n x-axis', fontsize=20)
    ax_poster_x.xaxis.set_major_locator(MaxNLocator(5))
    ax_poster_x.yaxis.set_major_locator(MaxNLocator(3))
    ax_poster_x.tick_params(axis='x', labelsize=10, direction='in')
    ax_poster_x.tick_params(axis='y', labelsize=10, direction='in')

    # y axis
    ax_poster_y.plot(
        poster_pano_t, poster_pano_vec[:, 1], color='red', linestyle='solid', lw=2.5)
    ax_poster_y.plot(
        poster_gt_t, poster_gt_vec[:, 1], color='black', linestyle='dashed', lw=2.5)
    ax_poster_y.plot(
        poster_glo_t, poster_glo_vec[:, 1], color='blue', linestyle='solid', lw=2.5)
    # ax_y.plot(pose_dr_t, pose_dr_y,
    #           color='yellow', linestyle='solid', lw=2.5)
    ax_poster_y.set_xlim(45, 55)
    ax_poster_y.set_ylim(-0.7, 0.5)
    ax_poster_y.set_title('y-axis', fontsize=20)
    ax_poster_y.xaxis.set_major_locator(MaxNLocator(5))
    ax_poster_y.yaxis.set_major_locator(MaxNLocator(3))
    ax_poster_y.tick_params(axis='x', labelsize=10, direction='in')
    ax_poster_y.tick_params(axis='y', labelsize=10, direction='in')

    # z axis
    ax_poster_z.plot(
        poster_pano_t, poster_pano_vec[:, 2], color='red', linestyle='solid', lw=2.5)
    ax_poster_z.plot(
        poster_gt_t, poster_gt_vec[:, 2], color='black', linestyle='dashed', lw=2.5)
    ax_poster_z.plot(
        poster_glo_t, poster_glo_vec[:, 2], color='blue', linestyle='solid', lw=2.5)
    # ax_z.plot(pose_dr_t, pose_dr_z,
    #           color='yellow', linestyle='solid', lw=2.5)
    ax_poster_z.set_xlim(45, 55)
    ax_poster_z.set_ylim(-1.4, 1.1)
    ax_poster_z.set_title('z-axis', fontsize=20)
    ax_poster_z.xaxis.set_major_locator(MaxNLocator(5))
    ax_poster_z.yaxis.set_major_locator(MaxNLocator(3))
    ax_poster_z.tick_params(axis='x', labelsize=10, direction='in')
    ax_poster_z.tick_params(axis='y', labelsize=10, direction='in')
    ax_poster_z.set_xlabel('time [s]', fontsize=15)


    # panorama rotation
    # x axis
    ax_panorama_x.plot(
        panorama_pano_t, panorama_pano_vec[:, 0], color='red', linestyle='solid', lw=2.5)
    ax_panorama_x.plot(
        panorama_gt_t, panorama_gt_vec[:, 0], color='black', linestyle='dashed', lw=2.5)
    ax_panorama_x.plot(
        panorama_glo_t, panorama_glo_vec[:, 0], color='blue', linestyle='solid', lw=2.5)
    # ax_x.plot(pose_dr_t, pose_dr_x,
    #           color='yellow', linestyle='solid', lw=2.5)
    ax_panorama_x.set_title('ESIM_panorama \n x-axis', fontsize=20)
    ax_panorama_x.xaxis.set_major_locator(MaxNLocator(5))
    ax_panorama_x.yaxis.set_major_locator(MaxNLocator(3))
    ax_panorama_x.tick_params(axis='x', labelsize=10, direction='in')
    ax_panorama_x.tick_params(axis='y', labelsize=10, direction='in')

    # y axis
    ax_panorama_y.plot(
        panorama_pano_t, panorama_pano_vec[:, 1], color='red', linestyle='solid', lw=2.5)
    ax_panorama_y.plot(
        panorama_gt_t, panorama_gt_vec[:, 1], color='black', linestyle='dashed', lw=2.5)
    ax_panorama_y.plot(
        panorama_glo_t, panorama_glo_vec[:, 1], color='blue', linestyle='solid', lw=2.5)
    # ax_y.plot(pose_dr_t, pose_dr_y,
    #           color='yellow', linestyle='solid', lw=2.5)
    ax_panorama_y.set_title('y-axis', fontsize=20)
    ax_panorama_y.xaxis.set_major_locator(MaxNLocator(5))
    ax_panorama_y.yaxis.set_major_locator(MaxNLocator(3))
    ax_panorama_y.tick_params(axis='x', labelsize=10, direction='in')
    ax_panorama_y.tick_params(axis='y', labelsize=10, direction='in')

    # z axis
    ax_panorama_z.plot(
        panorama_pano_t, panorama_pano_vec[:, 2], color='red', linestyle='solid', lw=2.5)
    ax_panorama_z.plot(
        panorama_gt_t, panorama_gt_vec[:, 2], color='black', linestyle='dashed', lw=2.5)
    ax_panorama_z.plot(
        panorama_glo_t, panorama_glo_vec[:, 2], color='blue', linestyle='solid', lw=2.5)
    # ax_z.plot(pose_dr_t, pose_dr_z,
    #           color='yellow', linestyle='solid', lw=2.5)
    ax_panorama_z.set_title('z-axis', fontsize=20)
    ax_panorama_z.xaxis.set_major_locator(MaxNLocator(5))
    ax_panorama_z.yaxis.set_major_locator(MaxNLocator(3))
    ax_panorama_z.tick_params(axis='x', labelsize=10, direction='in')
    ax_panorama_z.tick_params(axis='y', labelsize=10, direction='in')
    ax_panorama_z.set_xlabel('time [s]', fontsize=15)


    # legend
    lines, labels = ax_boxes_x.get_legend_handles_labels()
    fig.legend(lines, labels, ncol=3, 
               loc='lower center', fontsize=15)
    #fig.tight_layout()


    
    plt.show()


if __name__ == '__main__':
    main()
