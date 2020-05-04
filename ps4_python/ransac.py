import numpy as np
import random
import cv2


# Note: Not choosing parameters N and T, Brute Approach require N=100
def translation(line_pts, n_trials=100, tolerance=5):
    s = 1
    num_matches = len(line_pts)
    best_consensus = np.array([])
    best_tr = np.array([])
    for i in range(n_trials):
        rand_ind = random.sample(range(1, num_matches), s)
        pts = line_pts[rand_ind][:s]

        tr_mat = np.array([pts[0].x2 - pts[0].x1, pts[0].y2 - pts[0].y1], dtype=np.float64)

        curr_consensus = []
        for j in range(num_matches):
            cmp_mat = np.array([line_pts[j].x2 - line_pts[j].x1, line_pts[j].y2 - line_pts[j].y1], dtype=np.float64)

            if (abs(tr_mat[0] - cmp_mat[0]) < tolerance and abs(tr_mat[1] - cmp_mat[1]) < tolerance):
                curr_consensus.append(j)

        if len(curr_consensus) > len(best_consensus):
            best_consensus = np.copy(curr_consensus)
            best_tr = np.copy(tr_mat)

    return best_tr, line_pts[best_consensus]


def similarity(line_pts, n_trials=100, tolerance=5):
    s = 2
    num_matches = len(line_pts)
    best_consensus = np.array([])
    best_tr = np.array([])
    for i in range(n_trials):
        rand_ind = random.sample(range(1, num_matches), s)
        pts = line_pts[rand_ind][:s]

        j = 0
        A = np.array([[pts[j].x1, -pts[j].y1, 1, 0],
                      [pts[j].y1, pts[j].x1, 0, 1],
                      [pts[j + 1].x1, -pts[j + 1].y1, 1, 0],
                      [pts[j + 1].y1, pts[j + 1].x1, 0, 1]
                      ])
        b = np.array([pts[j].x2, pts[j].y2, pts[j + 1].x2, pts[j + 1].y2])
        tr_mat, _, _, _ = np.linalg.lstsq(A, b)
        tr_mat = np.array([[tr_mat[0], -tr_mat[1], tr_mat[2]],
                           [tr_mat[1], tr_mat[0], tr_mat[3]]
                           ])

        curr_consensus = []
        while j < (num_matches - 1):
            A = np.array([[line_pts[j].x1, -line_pts[j].y1, 1, 0],
                          [line_pts[j].y1, line_pts[j].x1, 0, 1],
                          [line_pts[j + 1].x1, -line_pts[j + 1].y1, 1, 0],
                          [line_pts[j + 1].y1, line_pts[j + 1].x1, 0, 1]
                          ])
            b = np.array([line_pts[j].x2, line_pts[j].y2, line_pts[j + 1].x2, line_pts[j + 1].y2])
            cmp_mat, _, _, _ = np.linalg.lstsq(A, b)
            cmp_mat = np.array([[cmp_mat[0], -cmp_mat[1], cmp_mat[2]],
                                [cmp_mat[1], cmp_mat[0], cmp_mat[3]]
                                ])
            if (np.all(abs(cmp_mat - tr_mat) < tolerance) == True):
                curr_consensus.append(j)
                j += 1
            j += 1

        if len(curr_consensus) > len(best_consensus):
            best_consensus = np.copy(curr_consensus)
            best_tr = np.copy(tr_mat)

    return best_tr, line_pts[best_consensus]


def affine(line_pts, n_trials=100, tolerance=5):
    s = 3
    num_matches = len(line_pts)
    best_consensus = np.array([])
    best_tr = np.array([])
    for i in range(n_trials):
        rand_ind = random.sample(range(1, num_matches), s)
        pts = line_pts[rand_ind][:s]

        src = np.array([[pts[0].x1, pts[0].y1], [pts[1].x1, pts[1].y1], [pts[2].x1, pts[2].y1]], dtype=np.float32)
        dest = np.array([[pts[0].x2, pts[0].y2], [pts[1].x2, pts[1].y2], [pts[2].x2, pts[2].y2]], dtype=np.float32)
        tr_mat = cv2.getAffineTransform(src, dest)
        j = 0
        curr_consensus = []
        while j < (num_matches-2):
            src = np.array([[line_pts[j].x1, line_pts[j].y1], [line_pts[j + 1].x1, line_pts[j + 1].y1],
                            [line_pts[j + 2].x1, line_pts[j + 2].y1]], dtype=np.float32)
            dest = np.array([[line_pts[j].x2, line_pts[j].y2], [line_pts[j + 1].x2, line_pts[j + 1].y2],
                             [line_pts[j + 2].x2, line_pts[j + 2].y2]], dtype=np.float32)
            cmp_mat = cv2.getAffineTransform(src, dest)

            if (np.all(abs(cmp_mat - tr_mat) < tolerance) == True):
                curr_consensus.append(j)
                curr_consensus.append(j + 1)
                curr_consensus.append(j + 2)
                j += 2
            j += 1
        if len(curr_consensus) > len(best_consensus):
            best_consensus = np.copy(curr_consensus)
            best_tr = np.copy(tr_mat)

    return best_tr, line_pts[best_consensus]
