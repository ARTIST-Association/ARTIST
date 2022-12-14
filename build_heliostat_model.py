import torch as th
import os

from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from yacs.config import CfgNode
from heliostat_models import AbstractHeliostat, Heliostat
from multi_nurbs_heliostat import MultiNURBSHeliostat, NURBSFacets
from nurbs_heliostat import AbstractNURBSHeliostat, NURBSHeliostat
import facets

def load_heliostat(
        cfg: CfgNode,
        sun_directions: th.Tensor,
        device: th.device,
) -> AbstractHeliostat:
    cp_path = os.path.expanduser(cfg.CP_PATH)
    cp = th.load(cp_path, map_location=device)
    if cfg.USE_NURBS:
        if 'facets' in cp:
            nurbs_heliostat_cls: Type[AbstractNURBSHeliostat] = \
                MultiNURBSHeliostat
        else:
            nurbs_heliostat_cls = NURBSHeliostat
        H: AbstractHeliostat = nurbs_heliostat_cls.from_dict(
            cp,
            device,
            nurbs_config=cfg.NURBS,
            config=cfg.H,
            receiver_center=cfg.AC.RECEIVER.CENTER,
            sun_directions=sun_directions,
        )
    else:
        H = Heliostat.from_dict(
            cp,
            device,
            receiver_center=cfg.AC.RECEIVER.CENTER,
            sun_directions=sun_directions,
        )
    return H


def _build_multi_nurbs_target(
        cfg: CfgNode,
        sun_directions: th.Tensor,
        device: th.device,
) -> MultiNURBSHeliostat:
    mnh_cfg = cfg.clone()
    mnh_cfg.defrost()
    mnh_cfg.H.SHAPE = 'Ideal'
    mnh_cfg.freeze()
    
    nurbs_cfg = mnh_cfg.NURBS.clone()
    nurbs_cfg.defrost()

    # We need this to get correct shapes.
    nurbs_cfg.SET_UP_WITH_KNOWLEDGE = True
    # Deactivate good-for-training options.
    nurbs_cfg.INITIALIZE_WITH_KNOWLEDGE = False
    nurbs_cfg.RECALCULATE_EVAL_POINTS = False
    nurbs_cfg.GROWING.INTERVAL = 0

    # Overwrite all attributes specified via `mnh_cfg.H.NURBS`.
    node_stack = [(nurbs_cfg, mnh_cfg.H.NURBS)]
    while node_stack:
        node, h_node = node_stack.pop()

        for attr in node.keys():
            if not hasattr(h_node, attr):
                continue

            if isinstance(getattr(node, attr), CfgNode):
                node_stack.append((
                    getattr(node, attr),
                    getattr(h_node, attr),
                ))
            else:
                setattr(node, attr, getattr(h_node, attr))

    nurbs_cfg.freeze()
    mnh = MultiNURBSHeliostat(
        mnh_cfg.H,
        nurbs_cfg,
        device,
        receiver_center=mnh_cfg.AC.RECEIVER.CENTER,
        sun_directions=sun_directions,
        setup_params=False,
    )

    assert isinstance(mnh.facets, NURBSFacets)
    for facet in mnh.facets:
        assert isinstance(facet, NURBSHeliostat)
        facet.set_ctrl_points(
            facet.ctrl_points
            + th.rand_like(facet.ctrl_points)
            * mnh_cfg.H.NURBS.MAX_ABS_NOISE
        )

    return mnh


def _multi_nurbs_to_standard(
        cfg: CfgNode,
        sun_directions: th.Tensor,
        mnh: MultiNURBSHeliostat,
) -> Heliostat:
    H = Heliostat(
        cfg.H,
        mnh.device,
        receiver_center=cfg.AC.RECEIVER.CENTER,
        sun_directions=sun_directions,
        setup_params=False,
    )
    discrete_points, normals = mnh.discrete_points_and_normals()

    H.facets = facets.Facets(
        H,
        mnh.facets.positions,
        mnh.facets.spans_n,
        mnh.facets.spans_e,
        mnh.facets.raw_discrete_points,
        mnh.facets.raw_discrete_points_ideal,
        mnh.facets.raw_normals,
        mnh.facets.raw_normals_ideal,
        mnh.facets.cant_rots,
    )
    H.params = mnh.nurbs_cfg
    H.height = mnh.height
    H.width = mnh.width
    H.rows = mnh.rows
    H.cols = mnh.cols

    return H


def build_target_heliostat(
        cfg: CfgNode,
        sun_directions: th.Tensor,
        device: th.device,
) -> Heliostat:
    print(cfg)
    if cfg.H.SHAPE.lower() == 'nurbs':
        mnh = _build_multi_nurbs_target(cfg, sun_directions, device)
        H = _multi_nurbs_to_standard(cfg, sun_directions, mnh)
    else:
        H = Heliostat(
            cfg.H,
            device,
            receiver_center=cfg.AC.RECEIVER.CENTER,
            sun_directions=sun_directions,
            setup_params=False,
        )
    return H

def build_heliostat(
        cfg: CfgNode,
        sun_directions: th.Tensor,
        device: th.device,
) -> AbstractHeliostat:
    if cfg.CP_PATH and os.path.isfile(os.path.expanduser(cfg.CP_PATH)):
        H = load_heliostat(cfg, sun_directions, device)
    else:
        if cfg.USE_NURBS:
            H = MultiNURBSHeliostat(
                cfg.H,
                cfg.NURBS,
                device,
                receiver_center=cfg.AC.RECEIVER.CENTER,
                sun_directions=sun_directions,
            )
        else:
            H = Heliostat(
                cfg.H,
                device,
                receiver_center=cfg.AC.RECEIVER.CENTER,
                sun_directions=sun_directions,
            )
    return H