import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def main():

    # load files
    try:
        vel_imu = np.loadtxt('data_gt/test_panorama/imu_angular.txt')
        vel_imu_t = vel_imu[:, 0]
        vel_imu_x = vel_imu[:, 1]
        vel_imu_y = vel_imu[:, 2]
        vel_imu_z = vel_imu[:, 3]

        vel_glo = np.loadtxt('data_globally/test_panorama/velocity.txt')
        vel_glo_t = vel_glo[:, 0]
        vel_glo_x = vel_glo[:, 1]
        vel_glo_y = vel_glo[:, 2]
        vel_glo_z = vel_glo[:, 3]

        vel_gt = np.loadtxt('data_gt/test_panorama/gt_angular.txt')
        vel_gt_t = vel_gt[:, 0]
        vel_gt_x = vel_gt[:, 1]
        vel_gt_y = vel_gt[:, 2]
        vel_gt_z = vel_gt[:, 3]

        pose_pano = np.loadtxt('data_panotracking/esim/estimated_pose.txt')

    except FileNotFoundError:
        print('Cannot open file!')
    except LookupError:
        print('Unkown unicode!')
    except UnicodeDecodeError:
        print('Decode error!')

    # try to calculate dvs_panotracking angular velocity
    row_num = pose_pano.shape[0]
    pose_pano = pose_pano[0:row_num, :]
    row_num = pose_pano.shape[0]
    print(pose_pano.shape[0])

    vel_pano = np.zeros((row_num-1, 4))
    for i in range(row_num-1):

        r1 = pose_pano[i, 1:4]
        r2 = pose_pano[i+1, 1:4]
        delta_t = pose_pano[i+1, 0]-pose_pano[i, 0]
        w = (r2-r1) / delta_t
        vel_pano[i, 1:4] = w

        # r1 = R.from_rotvec(pose_pano[i, 1:4])
        # r2 = R.from_rotvec(pose_pano[i+1, 1:4])

        # quat
        # delta_q = R.from_quat(r2.as_quat() - r1.as_quat())
        # w = 2*(delta_q*r1).as_quat()/(pose_pano[i+1, 0]-pose_pano[i, 0])
        # vel_pano[i, 1:4] = w[1:4]

        # matrix
        # s_w = ((r2*r1.inv()).as_matrix() - np.eye(3)) / \
        #     (pose_pano[i+1, 0]-pose_pano[i, 0])
        # w = np.array([s_w[2, 1], s_w[0, 2], s_w[1, 0]])
        # vel_pano[i, 1:4] = w

        vel_pano[i, 0] = pose_pano[i, 0]
    vel_pano_t = vel_pano[:, 0]
    vel_pano_x = -vel_pano[:, 1]
    vel_pano_y = -vel_pano[:, 3]
    vel_pano_z = -vel_pano[:, 2]

    # gt velocity compared with globally_alignment
    fig_aligned, ax_aligned = plt.subplots()
    ax_aligned.plot(vel_glo_t, vel_glo_x, 'b-', label='x_glo')
    ax_aligned.plot(vel_glo_t, vel_glo_y, 'y-', label='y_glo')
    ax_aligned.plot(vel_glo_t, vel_glo_z, 'g-', label='z_glo')
    ax_aligned.plot(vel_imu_t, vel_imu_x, 'b--', label='x_imu')
    ax_aligned.plot(vel_imu_t, vel_imu_y, 'y--', label='y_imu')
    ax_aligned.plot(vel_imu_t, vel_imu_z, 'g--', label='z_imu')
    ax_aligned.set_xlabel('time/s')
    ax_aligned.set_ylabel('rotation velocity [rad/s]')
    ax_aligned.set_title(
        'Globally_Aligned_Events angular velocity comparison with IMU')
    ax_aligned.legend()

    # gt velocity compared with dvs_pano
    fig_pano, ax_pano = plt.subplots()
    ax_pano.plot(vel_pano_t, vel_pano_x, 'b-', label='x_pano')
    ax_pano.plot(vel_pano_t, vel_pano_y, 'y-', label='y_pano')
    ax_pano.plot(vel_pano_t, vel_pano_z, 'g-', label='z_pano')
    ax_pano.plot(vel_imu_t, vel_imu_x, 'b--', label='x_imu')
    ax_pano.plot(vel_imu_t, vel_imu_y, 'y--', label='y_imu')
    ax_pano.plot(vel_imu_t, vel_imu_z, 'g--', label='z_imu')
    ax_pano.set_xlabel('time/s')
    ax_pano.set_ylabel('rotation velocity [rad/s]')
    ax_pano.set_title('Panotracking angular velocity comparison with IMU')
    ax_pano.legend()

    # get velocity compared with IMU and GT
    fig_gt, ax_gt = plt.subplots()
    ax_gt.plot(vel_gt_t, vel_gt_x, 'b--', label='x_gt')
    ax_gt.plot(vel_gt_t, vel_gt_y, 'y--', label='y_gt')
    ax_gt.plot(vel_gt_t, vel_gt_z, 'g--', label='z_gt')
    ax_gt.plot(vel_imu_t, vel_imu_x, 'b-', label='x_imu')
    ax_gt.plot(vel_imu_t, vel_imu_y, 'y-', label='y_imu')
    ax_gt.plot(vel_imu_t, vel_imu_z, 'g-', label='z_imu')
    ax_gt.set_xlabel('time/s')
    ax_gt.set_ylabel('rotation velocity [rad/s]')
    ax_gt.set_title('GT angular velocity comparison with IMU')
    ax_gt.legend()

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
