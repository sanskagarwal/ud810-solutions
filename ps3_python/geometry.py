import numpy as np

def lstsq(points_3d, points_2d):
    points = points_3d.shape[0]
    A = np.zeros((2*points, 11), dtype=np.float64)
    b = np.zeros((2*points, 1), dtype=np.float64)
    
    for i in range(points):
        X = points_3d[i,0]
        Y = points_3d[i,1]
        Z = points_3d[i,2]
        x = points_2d[i,0]
        y = points_2d[i,1]
        b[2*i] = x
        b[2*i+1] = y
        A[2*i] = [X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z]
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z]
        
    M, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
    M = np.append(M, 1)
    M = np.reshape(M, (3,4))
    
    return M

def lstsqF(pl, pr):
    points = pl.shape[0]
    A = np.zeros((points, 8), dtype=np.float64)
    b = np.zeros((points, 1), dtype=np.float64)
    
    for i in range(points):
        ul = pl[i,0]
        vl = pl[i,1]
        ur = pr[i,0]
        vr = pr[i,1]
        A[i] = [ul*ur, ur*vl, ur, vr*ul, vr*vl, vr, ul, vl]
        b[i] = -1
        
    F, res, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    F = np.append(F, 1)
    F = np.reshape(F, (3, 3))
    
    return F
