import torch
import typing

E = 0
N = 1
U = 2

# coordinate system identity
#  /1, 0, 0, 0\
# / 0, 1, 0, 0 \
# \ 0, 0, 1, 0 /
#  \0, 0, 0, 1/
def identity4x4(dtype: torch.dtype = torch.get_default_dtype(), device : torch.device = torch.device('cpu')) -> torch.Tensor:
    o = torch.ones(1, dtype=dtype, device=device, requires_grad=False)
    z = torch.zeros(1, dtype=dtype, device=device, requires_grad=False)

    cE =   torch.stack([o, z, z, z])
    cN =   torch.stack([z, o, z, z])
    cU =   torch.stack([z, z, o, z])
    cPOS = torch.stack([z, z, z, o])

    mat = torch.cat((cE, cN, cU, cPOS), dim=1)
    return mat

# coordinate system translation by ENU
#  /1, 0, 0, e\
# / 0, 1, 0, n \
# \ 0, 0, 1, u /
#  \0, 0, 0, 1/
def translation4x4(trans_vec : torch.Tensor, dtype: torch.dtype = torch.get_default_dtype(), device : torch.device = torch.device('cpu')) -> torch.Tensor:
    e, n, u = torch.split(trans_vec, 1)
    o = torch.ones(1, dtype=dtype, device=device, requires_grad=False)
    z = torch.zeros(1, dtype=dtype, device=device, requires_grad=False)

    cE =   torch.stack([o, z, z, z])
    cN =   torch.stack([z, o, z, z])
    cU =   torch.stack([z, z, o, z])
    cPOS = torch.stack([e, n, u, o])

    mat = torch.cat((cE, cN, cU, cPOS), dim=1)
    return mat

# right-handed coordinate system rotation around x axis (angle in rad)
#  /1, 0,    0, 0\
# / 0, ca, -sa, 0 \
# \ 0, sa,  ca, 0 /
#  \0, 0,    0, 1/
def eastRotation4x4(angle : torch.Tensor, dtype: torch.dtype = torch.get_default_dtype(), device : torch.device = torch.device('cpu')) -> torch.Tensor:
    s = torch.sin(angle)
    c = torch.cos(angle)
    o = torch.tensor(1, dtype=dtype, device=device, requires_grad=False)
    z = torch.tensor(0, dtype=dtype, device=device, requires_grad=False)

    rE =   torch.stack([o, z,  z, z])
    rN =   torch.stack([z, c, -s, z])
    rU =   torch.stack([z, s,  c, z])
    rPOS = torch.stack([z, z,  z, o])

    mat = torch.vstack((rE, rN, rU, rPOS))
    return mat

# right-handed coordinate system rotation around y axis (angle in rad)
#  /ca, 0, sa, 0\
# /  0, 1,  0, 0 \
# \-sa, 0, ca, 0 /
#  \ 0, 0,  0, 1/
def northRotation4x4(angle : torch.Tensor, dtype: torch.dtype = torch.get_default_dtype(), device : torch.device = torch.device('cpu')) -> torch.Tensor:
    s = torch.sin(angle)
    c = torch.cos(angle)
    o = torch.tensor(1, dtype=dtype, device=device, requires_grad=False)
    z = torch.tensor(0, dtype=dtype, device=device, requires_grad=False)

    rE =   torch.stack([c,  z, s, z])
    rN =   torch.stack([z,  o, z, z])
    rU =   torch.stack([-s, z, c, z])
    rPOS = torch.stack([z,  z, z, o])

    mat = torch.vstack((rE, rN, rU, rPOS))
    return mat

# right-handed coordinate system rotation around z axis (angle in rad)
#  /ca, -sa, 0, 0\
# / sa,  ca, 0, 0 \
# \  0,   0, 1, 0 /
#  \ 0,   0, 0, 1/
def upRotation4x4(angle : torch.Tensor, dtype: torch.dtype = torch.get_default_dtype(), device : torch.device = torch.device('cpu')) -> torch.Tensor:
    s = torch.sin(angle)
    c = torch.cos(angle)
    o = torch.tensor(1, dtype=dtype, device=device, requires_grad=False)
    z = torch.tensor(0, dtype=dtype, device=device, requires_grad=False)

    rE =   torch.stack([c, -s, z, z])
    rN =   torch.stack([s,  c, z, z])
    rU =   torch.stack([z,  z, o, z])
    rPOS = torch.stack([z,  z, z, o])

    mat = torch.vstack((rE, rN, rU, rPOS))
    return mat