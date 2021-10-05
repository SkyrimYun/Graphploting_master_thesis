import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import eye
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA
from matplotlib.ticker import MaxNLocator

def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def main():

    # load files
    try:
        glo = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/Globally_aligned_events_original/src/rotation_estimator/data/real_world/cross_360x2/position_rpg.txt')
        glo_t = glo[:, 0]
        glo_vec = R.from_quat(glo[:, 4:8]).as_rotvec()

        vel_imu = np.loadtxt(
            '/home/yunfan/work_spaces/master_thesis/datasets/cross_360x2/imu_angular.txt')
    except FileNotFoundError:
        print('Cannot open file!')
    except LookupError:
        print('Unkown unicode!')
    except UnicodeDecodeError:
        print('Decode error!')

    # try to calculate dead reckoning path from imu angular velocity
    dr = np.zeros((vel_imu.shape[0],4))
    dr[:,0]=vel_imu[:,0]
    dr[0, 1:4] = np.array([0,0,0])
    for i in range(1, dr.shape[0]):

        delta_t = (dr[i, 0]-dr[i-1, 0])
        start = R.from_rotvec(dr[i-1, 1:4])
        w = vel_imu[i-1, 1:4]

        rotvec_w = w*delta_t
        rotvec_s = start.inv().as_matrix().dot(rotvec_w)
        end = start * R.from_rotvec(rotvec_s)

        dr[i, 1:4] = end.as_rotvec()

    dr_t = dr[:, 0]
    dr_x = dr[:, 1]
    dr_y = dr[:, 2]
    dr_z = dr[:, 3]

    fig, (ax_x, ax_y, ax_z) = plt.subplots(3, figsize=[6, 12])

    # x axis
    ax_x.plot(dr_t,   dr_x, color='green',
              linestyle='solid', lw=2.5, label='IMU Dead Reckoning')
    ax_x.plot(glo_t,  glo_vec[:, 0], color='blue',
              linestyle='solid', lw=2.5, label='global aligment')
    ax_x.set_title('x-axis', fontsize=20)
    ax_x.xaxis.set_major_locator(MaxNLocator(5))
    ax_x.yaxis.set_major_locator(MaxNLocator(3))
    ax_x.tick_params(axis='x', labelsize=10, direction='in')
    ax_x.tick_params(axis='y', labelsize=10, direction='in')
    ax_x.set_ylabel('tilt [rad]', fontsize=25)

    # y axis
    ax_y.plot(dr_t,   dr_y, color='green', linestyle='solid', lw=2.5)
    ax_y.plot(glo_t,  glo_vec[:, 1], color='blue', linestyle='solid', lw=2.5)
    ax_y.set_title('y-axis', fontsize=20)
    ax_y.xaxis.set_major_locator(MaxNLocator(5))
    ax_y.yaxis.set_major_locator(MaxNLocator(3))
    ax_y.tick_params(axis='x', labelsize=10, direction='in')
    ax_y.tick_params(axis='y', labelsize=10, direction='in')
    ax_y.set_ylabel('pan [rad]', fontsize=25)

    # z axis
    ax_z.plot(dr_t,   dr_z, color='green', linestyle='solid', lw=2.5)
    ax_z.plot(glo_t,  glo_vec[:, 2], color='blue', linestyle='solid', lw=2.5)
    ax_z.set_title('z-axis', fontsize=20)
    ax_z.xaxis.set_major_locator(MaxNLocator(5))
    ax_z.yaxis.set_major_locator(MaxNLocator(3))
    ax_z.tick_params(axis='x', labelsize=10, direction='in')
    ax_z.tick_params(axis='y', labelsize=10, direction='in')
    ax_z.set_ylabel('roll [rad]', fontsize=25)
    ax_z.set_xlabel('time [s]', fontsize=25)

    # legend
    lines, labels = ax_x.get_legend_handles_labels()
    fig.legend(lines, labels, ncol=3,
               loc='lower center', fontsize=25)

    #fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
