import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import eye
from scipy.spatial.transform import Rotation as R
from numpy import linalg as LA
from matplotlib.ticker import MaxNLocator




def main():

    quat= np.array([-0.0344193, 0.56536, -0.0161267, 0.823968])
    vec = R.from_quat(quat).as_rotvec()
    print(vec)


if __name__ == '__main__':
    main()


