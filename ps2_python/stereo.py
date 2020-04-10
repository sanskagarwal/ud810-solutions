import numpy as np
import cv2

def ssd(patch, strip):
    tplCols = patch.shape[1]
    min_diff = np.inf
    best_x = tplCols//2
    search_range = strip.shape[1]
    for i in range(tplCols//2, search_range-tplCols//2):
        c_min = max(0, i-tplCols//2)
        c_max = min(search_range, i+tplCols//2+1)
        patch2 = strip[:, c_min:c_max]
        diff = np.sum((patch2-patch)**2)
        if(diff<min_diff):
            min_diff=diff
            best_x=i-tplCols//2
    return best_x
    
def disparity_ssd(L, R, kernel=(7,7), search_range=31):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    row, col = L.shape
    tplRows, tplCols = kernel
    D_L = np.zeros(L.shape, dtype=np.float64)
    for r in range(tplRows//2, row-tplRows//2):
        r_min = max(0, r-tplRows//2)
        r_max = min(row, r+tplRows//2+1)
        for c in range(tplCols//2, col-tplCols//2):
            c_min = max(0, c-tplCols//2)
            c_max = min(col, c+tplCols//2+1)
            
            cs_min = max(0, c_min-search_range//2)
            cs_max = min(col, c_max+search_range//2)
            L_patch = L[r_min:r_max, c_min:c_max]
            R_strip = R[r_min:r_max, cs_min:cs_max]
            min_loc = ssd(L_patch, R_strip)
            if(cs_min==0):
                D_L[r,c] = min_loc-c_min
            else:
                D_L[r,c] = min_loc-search_range//2

    return D_L
    
def disparity_ncorr(L, R, kernel=(7,7), search_range=31):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))
    
    Params:
    L: Grayscale left image
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """
    
    row, col = L.shape
    tplRows, tplCols = kernel
    D_L = np.zeros(L.shape, dtype=np.float64)
    for r in range(tplRows//2, row-tplRows//2):
        r_min = max(0, r-tplRows//2)
        r_max = min(row, r+tplRows//2+1)
        for c in range(tplCols//2, col-tplCols//2):
            c_min = max(0, c-tplCols//2)
            c_max = min(col, c+tplCols//2+1)
            
            cs_min = max(0, c_min-search_range//2)
            cs_max = min(col, c_max+search_range//2)
            L_patch = L[r_min:r_max, c_min:c_max]
            R_strip = R[r_min:r_max, cs_min:cs_max]
            res = cv2.matchTemplate(R_strip,L_patch,cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if(cs_min == 0):
                D_L[r,c] = max_loc[0]-c_min
            else:
                D_L[r,c] = max_loc[0]-search_range//2

    return D_L
        