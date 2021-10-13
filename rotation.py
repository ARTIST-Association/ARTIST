import math
import re

import torch as th

# Rotation functions ported from here:
# https://github.com/scipy/scipy/blob/50dbaf94a570fa817d0d72e0b94d9f87c4909a4e/scipy/spatial/transform/rotation.pyx


def rot_from_matrix(mat):
    is_single = False
    if (mat.ndim not in [2, 3] or mat.shape[-2:] != (3, 3)):
        raise ValueError(
            f'expected `mat` to have shape (3, 3) or (N, 3, 3), '
            f'got {mat.shape}'
        )

    if mat.shape == (3, 3):
        cmat = mat[None, :, :]
        is_single = True
    else:
        cmat = mat

    num_rots = cmat.shape[0]
    decision = th.empty(4, device=mat.device)

    quat = th.empty(num_rots, 4, device=mat.device)

    for ind in range(num_rots):
        decision[0] = cmat[ind, 0, 0]
        decision[1] = cmat[ind, 1, 1]
        decision[2] = cmat[ind, 2, 2]
        decision[3] = (
            cmat[ind, 0, 0]
            + cmat[ind, 1, 1]
            + cmat[ind, 2, 2]
        )
        choice = th.argmax(decision)

        if choice != 3:
            i = choice
            j = (i + 1) % 3
            k = (j + 1) % 3

            quat[ind, i] = 1 - decision[3] + 2 * cmat[ind, i, i]
            quat[ind, j] = cmat[ind, j, i] + cmat[ind, i, j]
            quat[ind, k] = cmat[ind, k, i] + cmat[ind, i, k]
            quat[ind, 3] = cmat[ind, k, j] - cmat[ind, j, k]
        else:
            quat[ind, 0] = cmat[ind, 2, 1] - cmat[ind, 1, 2]
            quat[ind, 1] = cmat[ind, 0, 2] - cmat[ind, 2, 0]
            quat[ind, 2] = cmat[ind, 1, 0] - cmat[ind, 0, 1]
            quat[ind, 3] = 1 + decision[3]

        quat[ind] /= th.linalg.norm(quat[ind])

    if is_single:
        return quat[0]
    else:
        return quat


def rot_from_rotvec(vec, degrees=False):
    is_single = False
    if degrees:
        vec = th.deg2rad(vec)

    if vec.ndim not in [1, 2] or vec.shape[-1] != 3:
        raise ValueError(
            f'expected `vec` to have shape (3,) or (N, 3), got {vec.shape}')

    if vec.shape == (3,):
        cvec = vec[None, :]
        is_single = True
    else:
        cvec = vec

    num_rots = cvec.shape[0]
    quat = th.empty(num_rots, 4, device=vec.device)

    for ind in range(num_rots):
        angle = th.linalg.norm(cvec[ind, :])

        if angle <= 1e-3:
            sqangle = angle * angle
            scale = 0.5 - sqangle / 48 + sqangle * sqangle / 3840
        else:
            scale = th.sin(angle / 2) / angle

        quat[ind, 0] = scale * cvec[ind, 0]
        quat[ind, 1] = scale * cvec[ind, 1]
        quat[ind, 2] = scale * cvec[ind, 2]
        quat[ind, 3] = th.cos(angle / 2)

    if is_single:
        return quat[0]
    else:
        return quat


def rot_is_single(rot):
    return rot.shape == (4,)


def rot_as_matrix(rot):
    is_single = rot_is_single(rot)
    if is_single:
        rot = rot[None, :]

    num_rots = rot.shape[0]
    mat = th.empty(num_rots, 3, 3, device=rot.device)

    for ind in range(num_rots):
        x = rot[ind, 0]
        y = rot[ind, 1]
        z = rot[ind, 2]
        w = rot[ind, 3]

        sqx = x * x
        sqy = y * y
        sqz = z * z
        sqw = w * w

        xy = x * y
        zw = z * w
        xz = x * z
        yw = y * w
        yz = y * z
        xw = x * w

        mat[ind, 0, 0] = sqx - sqy - sqz + sqw
        mat[ind, 1, 0] = 2 * (xy + zw)
        mat[ind, 2, 0] = 2 * (xz - yw)

        mat[ind, 0, 1] = 2 * (xy - zw)
        mat[ind, 1, 1] = -sqx + sqy - sqz + sqw
        mat[ind, 2, 1] = 2 * (yz + xw)

        mat[ind, 0, 2] = 2 * (xz + yw)
        mat[ind, 1, 2] = 2 * (yz - xw)
        mat[ind, 2, 2] = -sqx - sqy + sqz + sqw

    if is_single:
        return mat[0]
    else:
        return mat


def _elem_basis_vec(axis, device):
    vec = th.zeros(3, device=device)
    if axis == b'x':
        vec[0] = 1
    elif axis == b'y':
        vec[1] = 1
    elif axis == b'z':
        vec[2] = 1
    else:
        raise ValueError(f'unknown axis {axis}')
    return vec


def _compute_euler_from_matrix(mat, seq, extrinsic=False):
    if extrinsic:
        seq = seq[::-1]
    num_rots = mat.shape[0]

    n1 = _elem_basis_vec(seq[0:1], mat.device)
    n2 = _elem_basis_vec(seq[1:2], mat.device)
    n3 = _elem_basis_vec(seq[2:3], mat.device)

    sl = th.dot(th.cross(n1, n2), n3)
    cl = th.dot(n1, n3)

    offset = th.atan2(sl, cl)
    c = th.empty(3, 3, device=mat.device)
    c[0, :] = n2
    c[1, :] = th.cross(n1, n2)
    c[2, :] = n1

    rot = th.tensor([
        [1, 0, 0],
        [0, cl, sl],
        [0, -sl, cl],
    ], dtype=mat.dtype, device=mat.device)

    angles = th.empty(num_rots, 3, device=mat.device)
    eps = 1e-7

    for ind in range(num_rots):
        _angles = angles[ind, :]

        res = th.mm(c, mat[ind, :, :])
        matrix_trans = th.mm(res, c.T.mm(rot))

        matrix_trans[2, 2] = min(matrix_trans[2, 2], 1)
        matrix_trans[2, 2] = max(matrix_trans[2, 2], -1)
        _angles[1] = th.acos(matrix_trans[2, 2])

        safe1 = th.abs(_angles[1]) >= eps
        safe2 = th.abs(_angles[1] - math.pi) >= eps
        safe = safe1 and safe2

        _angles[1] += offset

        if safe:
            _angles[0] = th.atan2(matrix_trans[0, 2], -matrix_trans[1, 2])
            _angles[2] = th.atan2(matrix_trans[2, 0], matrix_trans[2, 1])

        if extrinsic:
            if not safe:
                _angles[0] = 0

            if not safe1:
                _angles[2] = th.atan2(matrix_trans[1, 0] - matrix_trans[0, 1],
                                      matrix_trans[0, 0] + matrix_trans[1, 1])

            if not safe2:
                _angles[2] = -th.atan2(matrix_trans[1, 0] + matrix_trans[0, 1],
                                       matrix_trans[0, 0] - matrix_trans[1, 1])

        else:
            if not safe:
                _angles[2] = 0

            if not safe1:
                _angles[0] = th.atan2(matrix_trans[1, 0] - matrix_trans[0, 1],
                                      matrix_trans[0, 0] + matrix_trans[1, 1])

            if not safe2:
                _angles[0] = th.atan2(matrix_trans[1, 0] + matrix_trans[0, 1],
                                      matrix_trans[0, 0] - matrix_trans[1, 1])

        if seq[0] == seq[2]:
            adjust = _angles[1] < 0 or _angles[1] > math.pi
        else:
            adjust = _angles[1] < -math.pi / 2 or _angles[1] > math.pi / 2

        if adjust and safe:
            _angles[0] += math.pi
            _angles[1] = 2 * offset - _angles[1]
            _angles[2] -= math.pi

        for i in range(3):
            if _angles[i] < -math.pi:
                _angles[i] += 2 * math.pi
            elif _angles[i] > math.pi:
                _angles[i] -= 2 * math.pi

        if extrinsic:
            tmp = _angles[0].clone()
            _angles[0] = _angles[2]
            _angles[2] = tmp

        if not safe:
            print('warning: gimbal lock detected; setting third angle to zero')

    return angles


def rot_as_euler(rot, seq, degrees=False):
    if len(seq) != 3:
        raise ValueError(f'expected 3 axes, got {seq}')

    intrinsic = re.match(r'^[XYZ]{1,3}$', seq) is not None
    extrinsic = re.match(r'^[xyz]{1,3}$', seq) is not None

    if not (intrinsic or extrinsic):
        raise ValueError(
            f'expected axes from `seq` to be from ["x", "y", "z"] or '
            f'["X", "Y", "Z"], got {seq}'
        )
    if any(seq[i] == seq[i + 1] for i in range(2)):
        raise ValueError(
            f'expected consecutive axes to be different, got {seq}')

    seq = seq.lower()
    mat = rot_as_matrix(rot)
    if mat.ndim == 2:
        mat = mat[None, :, :]
    angles = _compute_euler_from_matrix(mat, seq.encode(), extrinsic)
    if degrees:
        angles = th.rad2deg(angles)

    return angles[0] if rot_is_single(rot) else angles


def rot_apply(rot, vecs):
    mat = rot_as_matrix(rot)
    # mat: (3, 3)
    # vec: (1, 3)
    return th.squeeze(th.matmul(mat, vecs), 0)
