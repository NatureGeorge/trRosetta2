import numpy as np
import scipy
import scipy.spatial


def get_dihedrals(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    '''calculate dihedral angles defined by 4 sets of points'''

    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:, None]

    v = b0 - np.sum(b0*b1, axis=-1)[:, None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:, None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)

    return np.arctan2(y, x)


def get_angles(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    '''calculate planar angles defined by 3 sets of points'''

    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:, None]

    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:, None]

    x = np.sum(v*w, axis=1)

    return np.arccos(x)


def get_coords6d(xyz: np.ndarray, dmax: float = 20.0, dist_fill_value: float = 999.9):
    '''get 6d coordinates from x,y,z coords of N,Ca,C atoms'''

    nres = xyz.shape[1]

    # three anchor atoms
    N = xyz[0]
    Ca = xyz[1]
    C = xyz[2]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    # fast neighbors search to collect all
    # Cb-Cb pairs within dmax
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)

    # indices of contacting residues
    idx = np.array([[i, j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]
    mask_for_sym = idx0 > idx1
    idx0_sym = idx0[mask_for_sym]
    idx1_sym = idx1[mask_for_sym]

    # Cb-Cb distance matrix
    dist6d = np.full((nres, nres), dist_fill_value)
    dist6d[idx0_sym, idx1_sym] = dist6d[idx1_sym, idx0_sym] = np.linalg.norm(Cb[idx1_sym]-Cb[idx0_sym], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals (omega: -pi..pi)
    omega6d = np.zeros((nres, nres))
    omega6d[idx0_sym, idx1_sym] = omega6d[idx1_sym, idx0_sym] = get_dihedrals(Ca[idx0_sym], Cb[idx0_sym], Cb[idx1_sym], Ca[idx1_sym])

    # matrix of polar coord theta (theta: -pi..pi)
    theta6d = np.zeros((nres, nres))
    theta6d[idx0, idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # matrix of polar coord phi (phi: 0..pi)
    phi6d = np.zeros((nres, nres))
    phi6d[idx0, idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    return dist6d, omega6d, theta6d, phi6d
