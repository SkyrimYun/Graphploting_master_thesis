import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA


def main():

    # load files
    try:
        vel_imu = np.loadtxt(
            'data_gt/test_panorama/test_panorama3/imu_angular.txt')
        vel_imu_t = vel_imu[:, 0]
        vel_imu_x = vel_imu[:, 1]
        vel_imu_y = vel_imu[:, 2]
        vel_imu_z = vel_imu[:, 3]

        vel_gt = np.loadtxt(
            'data_gt/test_panorama/test_panorama3/gt_angular.txt')
        vel_gt_t = vel_gt[:, 0]
        vel_gt_x = vel_gt[:, 1]
        vel_gt_y = vel_gt[:, 2]
        vel_gt_z = vel_gt[:, 3]

        vel_glo = np.loadtxt(
            'data_globally/test_panorama/test_panorama2/velocity.txt')
        vel_glo_t = vel_glo[:, 0]
        vel_glo_x = vel_glo[:, 1]
        vel_glo_y = vel_glo[:, 2]
        vel_glo_z = vel_glo[:, 3]

        pose_pano = np.loadtxt(
            'data_panotracking/esim/test_panorama2/estimated_pose.txt')

        pose_gt = np.loadtxt(
            'data_gt/test_panorama/test_panorama3/pose_gt.txt')

    except FileNotFoundError:
        print('Cannot open file!')
    except LookupError:
        print('Unkown unicode!')
    except UnicodeDecodeError:
        print('Decode error!')

    # calculate dvs_panotracking angular velocity
    row_num = pose_gt.shape[0]
    pose_pano = pose_pano[0:row_num:50, :]
    row_num = pose_pano.shape[0]

    vel_pano = np.zeros((row_num-1, 4))
    for i in range(row_num-1):
        start = R.from_rotvec(pose_pano[i, 1:4])
        end = R.from_rotvec(pose_pano[i+1, 1:4])
        delta_t = pose_pano[i+1, 0]-pose_pano[i, 0]

        temp = (start.inv()*end).as_rotvec()
        vec_angle = LA.norm(temp)
        vec_axis = (temp / vec_angle).reshape(3, 1)

        o = start.as_matrix().dot(vec_axis)
        twist_rot = o*(vec_angle/delta_t)

        vel_pano[i, 1:4] = twist_rot.T
        vel_pano[i, 0] = pose_pano[i, 0]

    vel_pano_t = vel_pano[:, 0]
    vel_pano_x = -vel_pano[:, 1]
    vel_pano_y = -vel_pano[:, 2]
    vel_pano_z = -vel_pano[:, 3]

    # calculate GT angular velocity from GT pose
    row_num = pose_gt.shape[0]
    pose_gt = pose_gt[0:row_num:10, :]
    row_num = pose_gt.shape[0]

    vel_gt_pose = np.zeros((row_num-1, 4))
    for i in range(row_num-1):
        start = R.from_rotvec(pose_gt[i, 1:4])
        end = R.from_rotvec(pose_gt[i+1, 1:4])
        delta_t = pose_gt[i+1, 0]-pose_gt[i, 0]

        temp = (start.inv()*end).as_rotvec()
        vec_angle = LA.norm(temp)
        vec_axis = (temp / vec_angle).reshape(3, 1)

        o = start.as_matrix().dot(vec_axis)
        twist_rot = o*(vec_angle/delta_t)

        vel_gt_pose[i, 1:4] = twist_rot.T
        vel_gt_pose[i, 0] = pose_gt[i, 0]

    vel_gt_pose_t = vel_gt_pose[:, 0]
    vel_gt_pose_x = -vel_gt_pose[:, 1]
    vel_gt_pose_y = -vel_gt_pose[:, 2]
    vel_gt_pose_z = -vel_gt_pose[:, 3]

    # GT velocity compared with globally_alignment
    fig_aligned, ax_aligned = plt.subplots()
    ax_aligned.plot(vel_glo_t, vel_glo_x, 'b-', label='x_glo')
    ax_aligned.plot(vel_glo_t, vel_glo_y, 'y-', label='y_glo')
    ax_aligned.plot(vel_glo_t, vel_glo_z, 'g-', label='z_glo')
    ax_aligned.plot(vel_gt_t, vel_gt_x, 'b--', label='x_gt')
    ax_aligned.plot(vel_gt_t, vel_gt_y, 'y--', label='y_gt')
    ax_aligned.plot(vel_gt_t, vel_gt_z, 'g--', label='z_gt')
    ax_aligned.set_xlabel('time [s]')
    ax_aligned.set_ylabel('rotation velocity [rad/s]')
    ax_aligned.set_title(
        'Globally_Aligned_Events angular velocity comparison with GT')
    ax_aligned.legend()
    ax_aligned.set_xlim(0, 4)
    ax_aligned.set_ylim(-2, 2)

    # GT velocity compared with dvs_pano
    fig_pano, ax_pano = plt.subplots()
    ax_pano.plot(vel_pano_t, vel_pano_x, 'b-', label='x_pano')
    ax_pano.plot(vel_pano_t, vel_pano_y, 'y-', label='y_pano')
    ax_pano.plot(vel_pano_t, vel_pano_z, 'g-', label='z_pano')
    ax_pano.plot(vel_gt_t, vel_gt_x, 'b--', label='x_gt')
    ax_pano.plot(vel_gt_t, vel_gt_y, 'y--', label='y_gt')
    ax_pano.plot(vel_gt_t, vel_gt_z, 'g--', label='z_gt')
    ax_pano.set_xlabel('time [s]')
    ax_pano.set_ylabel('rotation velocity [rad/s]')
    ax_pano.set_title('Panotracking angular velocity comparison with GT')
    ax_pano.legend()
    ax_pano.set_xlim(0, 4)
    ax_pano.set_ylim(-2, 2)

    # GT velocity compared with IMU
    fig_gt, ax_gt = plt.subplots()
    ax_gt.plot(vel_gt_t, vel_gt_x, 'b--', label='x_gt')
    ax_gt.plot(vel_gt_t, vel_gt_y, 'y--', label='y_gt')
    ax_gt.plot(vel_gt_t, vel_gt_z, 'g--', label='z_gt')
    ax_gt.plot(vel_imu_t, vel_imu_x, 'b-', label='x_imu')
    ax_gt.plot(vel_imu_t, vel_imu_y, 'y-', label='y_imu')
    ax_gt.plot(vel_imu_t, vel_imu_z, 'g-', label='z_imu')
    ax_gt.set_xlabel('time [s]')
    ax_gt.set_ylabel('rotation velocity [rad/s]')
    ax_gt.set_title('GT angular velocity comparison with IMU')
    ax_gt.legend()
    ax_gt.set_xlim(0, 4)
    ax_gt.set_ylim(-2, 2)

    # GT velocity compared with GT velocity from pose
    fig_gt_pose, ax_gt_pose = plt.subplots()
    ax_gt_pose.plot(vel_gt_t, vel_gt_x, 'b--', label='x_gt')
    ax_gt_pose.plot(vel_gt_t, vel_gt_y, 'y--', label='y_gt')
    ax_gt_pose.plot(vel_gt_t, vel_gt_z, 'g--', label='z_gt')
    ax_gt_pose.plot(vel_gt_pose_t, vel_gt_pose_x, 'b-', label='x_gt_pose')
    ax_gt_pose.plot(vel_gt_pose_t, vel_gt_pose_y, 'y-', label='y_gt_pose')
    ax_gt_pose.plot(vel_gt_pose_t, vel_gt_pose_z, 'g-', label='z_gt_pose')
    ax_gt_pose.set_xlabel('time [s]')
    ax_gt_pose.set_ylabel('rotation velocity [rad/s]')
    ax_gt_pose.set_xlim(0, 4)
    ax_gt_pose.set_ylim(-2, 2)

    ax_gt_pose.set_title(
        'IMU angular velocity comparison with GT velocity from pose')
    ax_gt_pose.legend()

    # calculate angular velocity errors
    glo_error = np.zeros((vel_glo.shape[0], 3))
    for i in range(vel_glo.shape[0]):
        glo_t = vel_glo[i, 0]
        index_imu = np.argmin(np.abs(glo_t-vel_imu[:, 0]))
        glo_error[i, 0] = vel_glo[i, 1]-vel_imu[index_imu, 1]
        glo_error[i, 1] = vel_glo[i, 2]-vel_imu[index_imu, 2]
        glo_error[i, 2] = vel_glo[i, 3]-vel_imu[index_imu, 3]

    labels = ['x', 'y', 'z']

    fig_error_glo, ax_glo_error = plt.subplots()

    # rectangular box plot
    bplot1 = ax_glo_error.boxplot(glo_error,
                                  vert=True,  # vertical box alignment
                                  patch_artist=True,  # fill with color
                                  labels=labels)  # will be used to label x-ticks
    ax_glo_error.set_title('Angular velocity error: IMU with Globally-aligned')

    # fill with colors
    colors = ['pink', 'lightblue', 'lightgreen']

    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    # adding horizontal grid lines
    ax_glo_error.yaxis.grid(True)
    ax_glo_error.set_xlabel('axis')
    ax_glo_error.set_ylabel('Angular velocity error [rad/s]')

    plt.show()


if __name__ == '__main__':
    main()
