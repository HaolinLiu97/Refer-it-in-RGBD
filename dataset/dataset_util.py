import numpy as np

def rotate_aligned_boxes_along_axis(input_boxes, rot_mat, axis):
    centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
    new_centers = np.dot(centers, np.transpose(rot_mat))

    if axis == "x":
        d1, d2 = lengths[:, 1] / 2.0, lengths[:, 2] / 2.0
    elif axis == "y":
        d1, d2 = lengths[:, 0] / 2.0, lengths[:, 2] / 2.0
    else:
        d1, d2 = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0

    new_1 = np.zeros((d1.shape[0], 4))
    new_2 = np.zeros((d1.shape[0], 4))

    for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        crnrs = np.zeros((d1.shape[0], 3))
        crnrs[:, 0] = crnr[0] * d1
        crnrs[:, 1] = crnr[1] * d2
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_1[:, i] = crnrs[:, 0]
        new_2[:, i] = crnrs[:, 1]

    new_d1 = 2.0 * np.max(new_1, 1)
    new_d2 = 2.0 * np.max(new_2, 1)

    if axis == "x":
        new_lengths = np.stack((lengths[:, 0], new_d1, new_d2), axis=1)
    elif axis == "y":
        new_lengths = np.stack((new_d1, lengths[:, 1], new_d2), axis=1)
    else:
        new_lengths = np.stack((new_d1, new_d2, lengths[:, 2]), axis=1)

    return np.concatenate([new_centers, new_lengths], axis=1)

def rotx(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                    [0,  c,  -s],
                    [0,  s,  c]])

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0]<num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]