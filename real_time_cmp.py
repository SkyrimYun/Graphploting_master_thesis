import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA, sqrt


def main():

    
    pano_it=np.array([5,10,15,20,25])
    pano_packet_rmse_500=np.array([6.04,6.19,5.94,9.21,9.99])
    pano_packet_rmse_1000=np.array([5.87,5.87,5.94,5.85,5.91])
    pano_packet_rmse_2000=np.array([8.71,5.53,5.57,5.5,5.57])
    pano_packet_rmse_3000=np.array([30.03,4.81,5.76,4.8,4.89])
    pano_packet_rmse_4000=np.array([27.67,6.15,4.4,4.28,4.27])

    pano_packet_time_500=np.array([84.15,119.34,137.19,140.66,140.77])
    pano_packet_time_1000=np.array([49.79,69.39,70.01,101.18,101.79])
    pano_packet_time_2000=np.array([31.48,35.25,51.08,54.62,67.32])
    pano_packet_time_3000=np.array([23.71,34.17,43.07,46.00,56.02])
    pano_packet_time_4000 = np.array([19.76, 25.85, 33.83, 42.25, 51.72])


    # pano: RMSE
    fig_pano_rmse, ax_pano_rmse = plt.subplots()
    ax_pano_rmse.plot(pano_it, pano_packet_rmse_500, 'r-', label='500')
    ax_pano_rmse.plot(pano_it, pano_packet_rmse_1000, 'g-', label='1000')
    ax_pano_rmse.plot(pano_it, pano_packet_rmse_2000, 'b-', label='2000')
    ax_pano_rmse.plot(pano_it, pano_packet_rmse_3000, 'y-', label='3000')
    ax_pano_rmse.plot(pano_it, pano_packet_rmse_4000, 'c-', label='4000')

    # pano: time
    fig_pano_time, ax_pano_time = plt.subplots()
    ax_pano_time.plot(pano_it, pano_packet_time_500, 'r-', label='500')
    ax_pano_time.plot(pano_it, pano_packet_time_1000, 'g-', label='1000')
    ax_pano_time.plot(pano_it, pano_packet_time_2000, 'b-', label='2000')
    ax_pano_time.plot(pano_it, pano_packet_time_3000, 'y-', label='3000')
    ax_pano_time.plot(pano_it, pano_packet_time_4000, 'c-', label='4000')


    ax_pano_rmse.set_xlabel('iterations',fontsize=15)
    ax_pano_rmse.set_ylabel('RMSE [deg]',fontsize=15)
    ax_pano_time.set_xlabel('iterations',fontsize=15)
    ax_pano_time.set_ylabel('time [s]',fontsize=15)
    ax_pano_time.legend()
    fig_pano_rmse.tight_layout()
    fig_pano_time.tight_layout()


    glo_it=np.array([40,50,60,70,80])
    glo_dt_rmse_5 = np.array([12.09, 9.15, 7.67, 5.02, 4.99])
    glo_dt_rmse_10=np.array([6.30,5.82,6,5.24,4.83])
    glo_dt_rmse_15=np.array([37.36,11.92,4.86,4.62,5.25])
    glo_dt_rmse_20=np.array([72.83,41.85,12.24,5.09,4.72])
    glo_dt_rmse_25 = np.array([76.35, 76.94, 40.44, 17.26, 8.66])

    glo_dt_time_5=np.array([161.71,178.52,192.61,207.97,225.52])
    glo_dt_time_10=np.array([67.23,75.75,84.6,93.17,100.5])
    glo_dt_time_15=np.array([41.58,48.53,54.7,61.01,66.79])
    glo_dt_time_20=np.array([32.15,36.77,41.32,46.13,50.87])
    glo_dt_time_25 = np.array([26.37, 30.1, 33.54, 38.13, 42.04])


    # glo: RMSE
    fig_glo_rmse, ax_glo_rmse = plt.subplots()
    ax_glo_rmse.plot(glo_it, glo_dt_rmse_5, 'r-', label='5')
    ax_glo_rmse.plot(glo_it, glo_dt_rmse_10, 'g-', label='10')
    ax_glo_rmse.plot(glo_it, glo_dt_rmse_15, 'b-', label='15')
    ax_glo_rmse.plot(glo_it, glo_dt_rmse_20, 'y-', label='20')
    ax_glo_rmse.plot(glo_it, glo_dt_rmse_25, 'c-', label='25')

    # glo: time
    fig_glo_time, ax_glo_time = plt.subplots()
    ax_glo_time.plot(glo_it, glo_dt_time_5, 'r-', label='5')
    ax_glo_time.plot(glo_it, glo_dt_time_10, 'g-', label='10')
    ax_glo_time.plot(glo_it, glo_dt_time_15, 'b-', label='15')
    ax_glo_time.plot(glo_it, glo_dt_time_20, 'y-', label='20')
    ax_glo_time.plot(glo_it, glo_dt_time_25, 'c-', label='25')


    ax_glo_rmse.set_xlabel('iterations',fontsize=15)
    ax_glo_rmse.set_ylabel('RMSE [deg]',fontsize=15)
    ax_glo_time.set_xlabel('iterations',fontsize=15)
    ax_glo_time.set_ylabel('time [s]',fontsize=15)
    ax_glo_time.legend()
    fig_glo_rmse.tight_layout()
    fig_glo_time.tight_layout()


    plt.show()


if __name__ == '__main__':
    main()
