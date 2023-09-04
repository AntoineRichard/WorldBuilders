import numpy as np

def rpy2quat(rpy):
    q = np.zeros((rpy.shape[0], 4))
    sr = np.sin(rpy[:,0])
    cr = np.cos(rpy[:,0])
    sp = np.sin(rpy[:,1])
    cp = np.cos(rpy[:,1])
    sy = np.sin(rpy[:,2])
    cy = np.cos(rpy[:,2])
    q[:,0] = sr * cp * cy - cr * sp * sy
    q[:,1] = cr * sp * cy + sr * cp * sy
    q[:,2] = cr * cp * sy - sr * sp * cy
    q[:,3] = cr * cp * cy + sr * sp * sy

    return q