# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:38:11 2021

@author: parg_ma
"""

import math
import os
from typing import Callable, List, Optional, Tuple, TypeVar, Union

from matplotlib import cm
import torch
import torch as th
from yacs.config import CfgNode

import nurbs

# We would like to say that T can be everything but a list.
T = TypeVar('T')


def calculateSunAngles(hour,minute,sec,day,month,year,observerLatitude,observerLongitude):
    #in- and outputs are in degree    
    if (hour < 0 or hour > 23 or \
        minute < 0 or minute > 59 or \
        sec < 0 or sec > 59 or \
        day < 1 or day > 31 or \
        month < 1 or month > 12):
        raise ValueError("at least one value exeeded time range in calculateSunAngles")
    else:
        import math
        observerLatitudeInt = observerLatitude / 180.0*math.pi
        observerLongitudeInt = observerLongitude / 180.0*math.pi
        
        pressureInput = 1.01325 #Pressure in bar
        temperature = 20 #Temperature in °C
        
        UT = hour + minute / 60.0 + sec / 3600.0
        pressure = pressureInput / 1.01325
        delta_t = 0.0
        
        if month <= 2:
            dyear = year - 1.0
            dmonth = month + 12.0
        else:
            dyear = year
            dmonth = month
        
        trunc1 = math.floor(365.25*(dyear - 2000))
        trunc2 = math.floor(30.6001*(dmonth + 1))
        JD_t = trunc1 + trunc2 + day + UT / 24.0 - 1158.5
        t = JD_t + delta_t / 86400.0
        
        #standard JD and JDE (useless for the computation, they are computed for completeness)
        JDE = t + 2452640
        JD = JD_t + 2452640
        
        #HELIOCENTRIC LONGITUDE
        #linear increase + annual harmonic
        ang  = 0.0172019*t-0.0563
        heliocLongitude = 1.740940 + 0.017202768683*t + 0.0334118*math.sin(ang) + 0.0003488*math.sin(2.0*ang)
    
        #Moon perturbation
        heliocLongitude = heliocLongitude + 0.0000313*math.sin(0.2127730*t-0.585)
        #Harmonic correction
        heliocLongitude = heliocLongitude + 0.0000126*math.sin(0.004243*t+1.46) + \
                          0.0000235*math.sin(0.010727*t+0.72) + 0.0000276*math.sin(0.015799*t+2.35) + \
                          0.0000275*math.sin(0.021551*t-1.98) + 0.0000126*math.sin(0.031490*t-0.80)
                         
        #END HELIOCENTRIC LONGITUDE CALCULATION
        #Correction to longitude due to nutation
        t2 = t / 1000.0
        heliocLongitude = heliocLongitude + \
                          ((( -0.000000230796*t2 + 0.0000037976)*t2 - 0.000020458)*t2 + 0.00003976)*t2*t2
                          
        delta_psi = 0.0000833*math.sin(0.0009252*t - 1.173)
        
        #Earth axis inclination
        epsilon = -0.00000000621*t + 0.409086 + 0.0000446*math.sin(0.0009252*t + 0.397)
        #Geocentric global solar coordinates        
        geocSolarLongitude = heliocLongitude + math.pi +	delta_psi - 0.00009932
        
        s_lambda = math.sin(geocSolarLongitude)
        rightAscension = math.atan2(s_lambda*math.cos(epsilon),math.cos(geocSolarLongitude))
        
        declination = math.asin(math.sin(epsilon)*s_lambda)
        
        #local hour angle of the sun
        hourAngle = 6.30038809903*JD_t + 4.8824623 + delta_psi*0.9174 + observerLongitudeInt - rightAscension
        
        
        c_lat=math.cos(observerLatitudeInt)
        s_lat=math.sin(observerLatitudeInt)
        c_H=math.cos(hourAngle)
        s_H=math.sin(hourAngle)
        
        #Parallax correction to Right Ascension
        d_alpha = -0.0000426*c_lat*s_H
        topOCRightAscension=rightAscension+d_alpha;
        topOCHourAngle=hourAngle-d_alpha;
        
        #Parallax correction to Declination
        topOCDeclination = declination - 0.0000426*(s_lat-declination*c_lat)
        
        s_delta_corr=math.sin(topOCDeclination)
        c_delta_corr=math.cos(topOCDeclination)
        c_H_corr=c_H+d_alpha*s_H
        s_H_corr=s_H-d_alpha*c_H
        
        #Solar elevation angle, without refraction correction
        elevation_no_refrac = math.asin(s_lat*s_delta_corr + c_lat*c_delta_corr*c_H_corr)
        
        #Refraction correction: it is calculated only if elevation_no_refrac > elev_min
        elev_min = -0.01
        
        if (elevation_no_refrac > elev_min):
            refractionCorrection =	0.084217*pressure/(273.0+temperature)/math.tan(elevation_no_refrac+ 0.0031376/(elevation_no_refrac+0.089186))
        else:
            refractionCorrection = 0
        
        #elevationAngle = np.pi/2 - elevation_no_refrac - refractionCorrection;
        elevationAngle = elevation_no_refrac + refractionCorrection
        elevationAngle = elevationAngle * 180 / math.pi

        #azimuthAngle = math.atan2(s_H_corr, c_H_corr*s_lat - s_delta_corr/c_delta_corr*c_lat)
        azimuthAngle = -math.atan2(s_H_corr, c_H_corr*s_lat - s_delta_corr/c_delta_corr*c_lat)
        azimuthAngle = azimuthAngle * 180 / math.pi
        
    return azimuthAngle, elevationAngle



def get_sun_array(*datetime: list, **observer):
    """args must be in descending order years,months,days..."""
    years = [2021]
    months = [6]
    days = [21]
    hours = list(range(6,19))
    minutes = [0,30]
    secs = [0]   
    
    num_args = len(datetime)
    if num_args == 0:
        print("generate values for 21.06.2021")
    if num_args >= 1:
        years = datetime[0]
        if num_args >= 2:
            months = datetime[1]
            if num_args >= 3:
                days  = datetime[2]
                if num_args >= 4:
                    hours = datetime[3]
                    if num_args >= 5:
                        minutes  = datetime[4]
                        if num_args >= 6:
                            secs = datetime[5]

    observerLatitude = observer.get('latitude',  50.92)
    observerLongitude = observer.get('longitude', 6.36)

    # sunAngles = np.empty((3,1440,2))
    extras = []
    ae = []
    for year in years:
        for month in months:
            for day in days:
                for hour in hours:
                    for minute in minutes:
                        for sec in secs:
                            azi, ele = calculateSunAngles(hour,minute,sec,day,month,year,observerLatitude,observerLongitude)
                            extras.append([year,month,day,hour,minute,sec,azi,ele])
                            ae.append([azi,ele])
    ae = th.tensor(ae)
    sun_vecs = ae_to_vec(ae[:,0], ae[:,1])
    return sun_vecs, extras


def axis_angle_rotation(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    angle = th.tensor(angle_rad, dtype=axis.dtype, device=axis.device)
    cos = th.cos(angle)
    icos = 1 - cos
    sin = th.sin(angle)
    x = axis[..., 0]
    y = axis[..., 1]
    z = axis[..., 2]
    axis_sq = axis**2

    rows = [
        th.stack(row, dim=-1)
        for row in [
                [
                    cos + axis_sq[..., 0] * icos,
                    x * y * icos - z * sin,
                    x * z * icos + y * sin,
                ],
                [
                    y * x * icos + z * sin,
                    cos + axis_sq[..., 1] * icos,
                    y * z * icos - x * sin,
                ],
                [
                    z * x * icos - y * sin,
                    z * y * icos + x * sin,
                    cos + axis_sq[..., 2] * icos,
                ],
        ]
    ]
    return th.stack(rows, dim=1)


def deflec_facet_zs(
        points: torch.Tensor,
        normals: torch.Tensor,
) -> torch.Tensor:
    """Calculate z values for a surface given by normals at x-y-planar
    positions.

    We are left with a different unknown offset for each z value; we
    assume this to be constant.
    """
    distances = horizontal_distance(
        points.unsqueeze(0),
        points.unsqueeze(1),
    )
    distances, closest_indices = distances.sort(dim=-1)
    del distances
    # Take closest point that isn't the point itself.
    closest_indices = closest_indices[..., 1]

    midway_normal = normals + normals[closest_indices]
    midway_normal /= th.linalg.norm(midway_normal, dim=-1, keepdims=True)

    rot_z_90deg = th.tensor(
        [
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1],
        ],
        dtype=points.dtype,
        device=points.device,
    )

    connector = points[closest_indices] - points
    connector_norm = th.linalg.norm(connector, dim=-1)
    orthogonal = th.matmul(rot_z_90deg, connector.T).T
    orthogonal /= th.linalg.norm(orthogonal, dim=-1, keepdims=True)
    tilted_connector = th.cross(orthogonal, midway_normal)
    tilted_connector /= th.linalg.norm(tilted_connector, dim=-1, keepdims=True)

    angle = th.acos(th.clamp(
        (
            batch_dot(tilted_connector, connector).squeeze(-1)
            / connector_norm
        ),
        -1,
        1,
    ))
    zs = connector_norm * th.tan(angle)

    return zs


def _all_angles(points, normals, closest_indices, remaining_indices):
    connector = (points[closest_indices] - points).unsqueeze(1)
    other_connectors = (
        points[remaining_indices]
        - points.unsqueeze(1)
    )
    angles = th.acos(th.clamp(
        (
            batch_dot(connector, other_connectors).squeeze(-1)
            / (
                th.linalg.norm(connector, dim=-1)
                * th.linalg.norm(other_connectors, dim=-1)
            )
        ),
        -1,
        1,
    )).squeeze(-1)

    # Give the angles a rotation direction.
    angles *= (
        1
        - 2 * (
            batch_dot(
                normals.unsqueeze(1),
                # Cross product does not support broadcasting, so do it
                # manually.
                th.cross(
                    th.tile(connector, (1, other_connectors.shape[1], 1)),
                    other_connectors,
                    dim=-1,
                ),
            ).squeeze(-1)
            < 0
        )
    )

    # And convert to 360° rotations.
    tau = 2 * th.tensor(math.pi, dtype=angles.dtype, device=angles.device)
    angles = th.where(angles < 0, tau + angles, angles)
    return angles


def _find_angles_in_other_slices(angles, num_slices):
    dtype = angles.dtype
    device = angles.device
    # Set up uniformly sized cake/pizza slices for which to find angles.
    tau = 2 * th.tensor(math.pi, dtype=dtype, device=device)
    angle_slice = tau / num_slices

    angle_slices = (
        th.arange(
            num_slices,
            dtype=dtype,
            device=device,
        )
        * angle_slice
    ).unsqueeze(-1).unsqueeze(-1)
    # We didn't calculate angles in the "zeroth" slice so we disregard them.
    angle_start = angle_slices[1:] - angle_slice / 2
    angle_end = angle_slices[1:] + angle_slice / 2

    # Find all angles lying in each slice.
    angles_in_slice = ((angle_start <= angles) & (angles < angle_end))
    return angles_in_slice, angle_slices


def deflec_facet_zs_many(
        points: torch.Tensor,
        normals: torch.Tensor,
        ideal_normals: torch.Tensor,
        num_samples: int = 4,
        use_weighted_average: bool = False,
        eps: float = 1e-6,
) -> torch.Tensor:
    """Calculate z values for a surface given by normals at x-y-planar
    positions.

    We are left with a different unknown offset for each z value; we
    assume this to be constant.
    """
    # TODO When `num_samples == 1`, we can just use the old method.
    device = points.device
    dtype = points.dtype

    distances = horizontal_distance(
        points.unsqueeze(0),
        points.unsqueeze(1),
    )
    distances, distance_sorted_indices = distances.sort(dim=-1)
    del distances
    # Take closest point that isn't the point itself.
    closest_indices = distance_sorted_indices[..., 1]

    # Take closest point in different directions from the given point.

    # For that, first calculate angles between direction to closest
    # point and all others, sorted by distance.
    angles = _all_angles(
        points,
        normals,
        closest_indices,
        distance_sorted_indices[..., 2:],
    ).unsqueeze(0)

    # Find positions of all angles in each slice except the zeroth one.
    angles_in_slice, angle_slices = _find_angles_in_other_slices(
        angles, num_samples)

    # And take the first one.angle we found in each slice. Remember
    # these are still sorted by distance, so we obtain the first
    # matching angle that is also closest to the desired point.
    #
    # We need to handle not having any slices except the zeroth one
    # extra.
    if len(angles_in_slice) > 1:
        angle_indices = th.argmax(angles_in_slice.long(), dim=-1)
    else:
        angle_indices = th.empty(
            (0, len(points)), dtype=th.long, device=device)

    # Select the angles we found for each slice.
    angles = th.gather(angles.squeeze(0), -1, angle_indices.T)

    # Handle _not_ having found an angle. We here create an array of
    # booleans, indicating whether we found an angle, for each slice.
    found_angles = th.gather(
        angles_in_slice,
        -1,
        angle_indices.unsqueeze(-1),
    ).squeeze(-1)
    # We always found something in the zeroth slice, so add those here.
    found_angles = th.cat([
        th.ones((1,) + found_angles.shape[1:], dtype=th.bool, device=device),
        found_angles,
    ], dim=0)
    del angles_in_slice

    # Set up some numbers for averaging.
    if use_weighted_average:
        angle_diffs = (
            th.cat([
                th.zeros((len(angles), 1), dtype=dtype, device=device),
                angles,
            ], dim=-1)
            - angle_slices.squeeze(-1).T
        )
        # Inverse difference in angle.
        weights = 1 / (angle_diffs + eps).T
        del angle_diffs
    else:
        # Number of samples we found angles for.
        num_available_samples = th.count_nonzero(found_angles, dim=0)

    # Finally, combine the indices of the closest points (zeroth slice)
    # with the indices of all closest points in the other slices.
    closest_indices = th.cat((
        closest_indices.unsqueeze(0),
        angle_indices,
    ), dim=0)
    del angle_indices, angle_slices

    midway_normal = normals + normals[closest_indices]
    midway_normal /= th.linalg.norm(midway_normal, dim=-1, keepdims=True)

    rot_90deg = axis_angle_rotation(ideal_normals, math.pi / 2)

    connector = points[closest_indices] - points
    connector_norm = th.linalg.norm(connector, dim=-1)
    orthogonal = th.matmul(
        rot_90deg.unsqueeze(0),
        connector.unsqueeze(-1),
    ).squeeze(-1)
    orthogonal /= th.linalg.norm(orthogonal, dim=-1, keepdims=True)
    tilted_connector = th.cross(orthogonal, midway_normal, dim=-1)
    tilted_connector /= th.linalg.norm(tilted_connector, dim=-1, keepdims=True)
    tilted_connector *= th.sign(connector[..., -1]).unsqueeze(-1)

    angle = th.acos(th.clamp(
        (
            batch_dot(tilted_connector, connector).squeeze(-1)
            / connector_norm
        ),
        -1,
        1,
    ))
    # Here, we handle values for which we did not find an angle. For
    # some reason, the NaNs those create propagate even to supposedly
    # unaffected values, so we handle them explicitly.
    angle = th.where(
        found_angles & ~th.isnan(angle),
        angle,
        th.tensor(0.0, dtype=dtype, device=device),
    )

    # Average over each slice.
    if use_weighted_average:
        zs = (
            (weights * connector_norm * th.tan(angle)).sum(dim=0)
            / (weights * found_angles.to(dtype)).sum(dim=0)
        )
    else:
        zs = (
            (connector_norm * th.tan(angle)).sum(dim=0)
            / num_available_samples
        )

    return zs


def with_outer_list(values: Union[List[T], List[List[T]]]) -> List[List[T]]:
    # Type errors come from T being able to be a list. So we ignore them
    # as "type negation" ("T can be everything except a list") is not
    # currently supported.

    if isinstance(values[0], list):
        return values  # type: ignore[return-value]
    return [values]  # type: ignore[list-item]


def vec_to_ae(vec: torch.Tensor) -> torch.Tensor:
    """
    converts ENU vector to azimuth, elevation

    Parameters
    ----------
    vec : tensor (N,3)
        Batch of N spherical vectors

    Returns
    -------
    tensor
        returns Azi, Ele in ENU coordsystem

    """
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    device = vec.device

    north = th.tensor([0, 1, 0], dtype=th.get_default_dtype(), device=device)
    up = th.tensor([0, 0, 1], dtype=th.get_default_dtype(), device=device)

    xy_plane = vec.clone()
    xy_plane[:, 2] = 0
    xy_plane = xy_plane / th.linalg.norm(xy_plane, dim=1).unsqueeze(1)

    a = -th.rad2deg(th.arccos(th.matmul(xy_plane, north)))
    a = th.where(vec[:, 0] < 0, a, -a)

    e = -(th.rad2deg(th.arccos(th.matmul(vec, up))) - 90)
    return th.stack([a, e], dim=1)


def ae_to_vec(
        az: torch.Tensor,
        el: torch.Tensor,
        srange: float = 1.0,
        deg: bool = True,
) -> torch.Tensor:
    """
    Azimuth, Elevation, Slant range to target to East, North, Up

    Parameters
    ----------
    azimuth : float
            azimuth clockwise from north (degrees)
    elevation : float
        elevation angle above horizon, neglecting aberrations (degrees)
    srange : float
        slant range [meters]
    deg : bool, optional
        degrees input/output  (False: radians in/out)

    Returns
    --------
    e : float
        East ENU coordinate (meters)
    n : float
        North ENU coordinate (meters)
    u : float
        Up ENU coordinate (meters)
    """
    if deg:
        el = th.deg2rad(el)
        az = th.deg2rad(az)

    r = srange * th.cos(el)

    rot_vec = th.stack(
        [r * th.sin(az), r * th.cos(az), srange * th.sin(el)],
        dim=1,
    )
    return rot_vec


def colorize(
        image_tensor: torch.Tensor,
        colormap: str = 'jet',
) -> torch.Tensor:
    """

    Parameters
    ----------
    image_tensor : tensor
        expects tensor of shape [H,W]
    colormap : string, optional
        choose_colormap. The default is 'jet'.

    Returns
    -------
    colored image tensor of CHW

    """
    image_tensor = image_tensor.clone() / image_tensor.max()
    prediction_image = image_tensor.squeeze().detach().cpu().numpy()

    color_map = cm.get_cmap('jet')
    mapped_image = th.tensor(color_map(prediction_image)).permute(2, 1, 0)
    # mapped_image8 = (255*mapped_image).astype('uint8')
    # print(colored_prediction_image.shape)

    return mapped_image


def load_config_file(
        cfg: CfgNode,
        config_file_loc: str,
        experiment_name: Optional[str] = None,
) -> CfgNode:
    if len(os.path.splitext(config_file_loc)[1]) == 0:
        config_file_loc += '.yaml'
    cfg.merge_from_file(config_file_loc)
    if experiment_name:
        cfg.merge_from_list(["NAME", experiment_name])
    cfg.freeze()
    return cfg


def flatten_aimpoints(aimpoints: torch.Tensor) -> torch.Tensor:
    X = th.flatten(aimpoints[0])
    Y = th.flatten(aimpoints[1])
    Z = th.flatten(aimpoints[2])
    aimpoints = th.stack((X, Y, Z), dim=1)
    return aimpoints


def curl(
        f: Callable[[torch.Tensor], torch.Tensor],
        arg: torch.Tensor,
) -> torch.Tensor:
    jac = th.autograd.functional.jacobian(f, arg, create_graph=True)

    rot_x = jac[2][1] - jac[1][2]
    rot_y = jac[0][2] - jac[2][0]
    rot_z = jac[1][0] - jac[0][1]

    return th.tensor([rot_x, rot_y, rot_z])


def find_larger_divisor(num: int) -> int:
    divisor = int(th.sqrt(th.tensor(num)))
    while num % divisor != 0:
        divisor += 1
    return divisor


def find_perpendicular_pair(
        base_vec: torch.Tensor,
        vecs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half_pi = th.tensor(math.pi, device=vecs.device) / 2
    for vec_x in vecs[1:]:
        surface_direction_x = vec_x - base_vec
        surface_direction_x /= th.linalg.norm(surface_direction_x)
        for vec_y in vecs[2:]:
            surface_direction_y = vec_y - base_vec
            surface_direction_y /= th.linalg.norm(surface_direction_y)
            if th.isclose(
                    th.acos(th.dot(
                        surface_direction_x,
                        surface_direction_y,
                    )),
                    half_pi,
            ):
                return surface_direction_x, surface_direction_y
    raise ValueError('could not calculate surface normal')


def _cartesian_linspace_around(
        minval_x: Union[float, torch.Tensor],
        maxval_x: Union[float, torch.Tensor],
        num_x: int,
        minval_y: Union[float, torch.Tensor],
        maxval_y: Union[float, torch.Tensor],
        num_y: int,
        device: th.device,
        dtype: Optional[th.dtype] = None,
):
    if dtype is None:
        dtype = th.get_default_dtype()
    if not isinstance(minval_x, th.Tensor):
        minval_x = th.tensor(minval_x, dtype=dtype, device=device)
    if not isinstance(maxval_x, th.Tensor):
        maxval_x = th.tensor(maxval_x, dtype=dtype, device=device)
    if not isinstance(minval_y, th.Tensor):
        minval_y = th.tensor(minval_y, dtype=dtype, device=device)
    if not isinstance(maxval_y, th.Tensor):
        maxval_y = th.tensor(maxval_y, dtype=dtype, device=device)
    spline_max = 1

    minval_x = minval_x.clamp(0, spline_max)
    maxval_x = maxval_x.clamp(0, spline_max)
    minval_y = minval_y.clamp(0, spline_max)
    maxval_y = maxval_y.clamp(0, spline_max)

    points_x = th.linspace(
        minval_x, maxval_x, num_x, device=device)  # type: ignore[arg-type]
    points_y = th.linspace(
        minval_y, maxval_y, num_y, device=device)  # type: ignore[arg-type]
    points = th.cartesian_prod(points_x, points_y)
    return points


# TODO choose uniformly between spans (not super important
#      as our knots are uniform as well)
def initialize_spline_eval_points(
        rows: int,
        cols: int,
        device: th.device,
) -> torch.Tensor:
    return _cartesian_linspace_around(0, 1, rows, 0, 1, cols, device)


def initialize_spline_eval_points_perfectly(
        points: torch.Tensor,
        degree_x: int,
        degree_y: int,
        ctrl_points: torch.Tensor,
        ctrl_weights: torch.Tensor,
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
) -> torch.Tensor:
    eval_points, distances = nurbs.invert_points_slow(
            points,
            degree_x,
            degree_y,
            ctrl_points,
            ctrl_weights,
            knots_x,
            knots_y,
    )
    return eval_points


def round_positionally(x: torch.Tensor) -> torch.Tensor:
    """Round usually but round .5 decimal point depending on position.

    If the decimal point is .5, values in the lower half of `x` are
    rounded down while values in the upper half of `x` are rounded up.

    The halfway point is obtained by rounding up.
    """
    x_middle = int(th.tensor(len(x) / 2).round())

    # Round lower values down, upper values up.
    # This makes the indices become mirrored around the middle
    # index.
    lower_half = x[:x_middle]
    upper_half = x[x_middle:]
    point_five = th.tensor(0.5, device=x.device)

    lower_half = th.where(
        th.isclose(lower_half % 1, point_five),
        lower_half.floor(),
        lower_half,
    ).long()
    upper_half = th.where(
        th.isclose(upper_half % 1, point_five),
        upper_half.ceil(),
        upper_half,
    ).long()

    x = th.cat([lower_half, upper_half])
    return x


def horizontal_distance(
        a: torch.Tensor,
        b: torch.Tensor,
        ord: Union[int, float, str] = 2,
) -> torch.Tensor:
    return th.linalg.norm(b[..., :-1] - a[..., :-1], dim=-1, ord=ord)


def distance_weighted_avg(
        distances: torch.Tensor,
        points: torch.Tensor,
) -> torch.Tensor:
    # Handle distances of 0 just in case with a very small value.
    distances = th.where(
        distances == 0,
        th.tensor(
            th.finfo(distances.dtype).tiny,
            device=distances.device,
            dtype=distances.dtype,
        ),
        distances,
    )
    inv_distances = 1 / distances.unsqueeze(-1)
    weighted = inv_distances * points
    total = weighted.sum(dim=-2)
    total = total / inv_distances.sum(dim=-2)
    return total


def calc_knn_averages(
        points: torch.Tensor,
        neighbours: torch.Tensor,
        k: int,
) -> torch.Tensor:
    distances = horizontal_distance(
        points.unsqueeze(1),
        neighbours.unsqueeze(0),
    )
    distances, closest_indices = distances.sort(dim=-1)
    distances = distances[..., :k]
    closest_indices = closest_indices[..., :k]

    averaged = distance_weighted_avg(distances, neighbours[closest_indices])
    return averaged


def initialize_spline_ctrl_points(
        control_points: torch.Tensor,
        origin: torch.Tensor,
        rows: int,
        cols: int,
        h_width: float,
        h_height: float,
):
    device = control_points.device
    origin_offsets_x = th.linspace(
        -h_width / 2, h_width / 2, rows, device=device)
    origin_offsets_y = th.linspace(
        -h_height / 2, h_height / 2, cols, device=device)
    origin_offsets = th.cartesian_prod(origin_offsets_x, origin_offsets_y)
    origin_offsets = th.hstack((
        origin_offsets,
        th.zeros((len(origin_offsets), 1), device=device),
    ))
    control_points[:] = (origin + origin_offsets).reshape(control_points.shape)


def calc_closest_ctrl_points(
        control_points: torch.Tensor,
        world_points: torch.Tensor,
        k: int = 4,
) -> torch.Tensor:
    new_control_points = calc_knn_averages(
        control_points.reshape(-1, control_points.shape[-1]),
        world_points,
        k,
    )
    return new_control_points.reshape(control_points.shape)


def _make_structured_points_from_corners(
        points: torch.Tensor,
        rows: int,
        cols: int,
) -> Tuple[torch.Tensor, int, int]:
    x_vals = points[:, 0]
    y_vals = points[:, 1]

    x_min = x_vals.min()
    x_max = x_vals.max()
    y_min = y_vals.min()
    y_max = y_vals.max()

    x_vals = th.linspace(
        x_min, x_max, rows, device=x_vals.device)  # type: ignore[arg-type]
    y_vals = th.linspace(
        y_min, y_max, cols, device=y_vals.device)  # type: ignore[arg-type]

    structured_points = th.cartesian_prod(x_vals, y_vals)

    distances = th.linalg.norm(
        (
            structured_points.unsqueeze(1)
            - points[:, :-1].unsqueeze(0)
        ),
        dim=-1,
    )
    argmin_distances = distances.argmin(1)
    z_vals = points[argmin_distances, -1]
    structured_points = th.cat(
        [structured_points, z_vals.unsqueeze(-1)], dim=-1)

    return structured_points, rows, cols


def _make_structured_points_from_unique(
        points: torch.Tensor,
        tolerance: float,
) -> Tuple[torch.Tensor, int, int]:
    x_vals = points[:, 0]
    x_vals = th.unique(x_vals, sorted=True)

    prev_i = 0
    keep_indices = [prev_i]
    for (i, x) in enumerate(x_vals[1:]):
        if not th.isclose(x, x_vals[prev_i], atol=tolerance):
            prev_i = i
            keep_indices.append(prev_i)
    x_vals = x_vals[keep_indices]

    y_vals = points[:, 0]
    y_vals = th.unique(y_vals, sorted=True)

    prev_i = 0
    keep_indices = [prev_i]
    for (i, y) in enumerate(y_vals[1:]):
        if not th.isclose(y, y_vals[prev_i], atol=tolerance):
            prev_i = i
            keep_indices.append(prev_i)
    y_vals = y_vals[keep_indices]

    structured_points = th.cartesian_prod(x_vals, y_vals)

    distances = th.linalg.norm(
        (
            structured_points.unsqueeze(1)
            - points[:, :-1].unsqueeze(0)
        ),
        dim=-1,
    )
    argmin_distances = distances.argmin(1)
    z_vals = points[argmin_distances, -1]
    structured_points = th.cat(
        [structured_points, z_vals.unsqueeze(-1)], dim=-1)

    rows = len(x_vals)
    cols = len(y_vals)
    return structured_points, rows, cols


def make_structured_points(
        points: torch.Tensor,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        tolerance: float = 0.0075,
) -> Tuple[torch.Tensor, int, int]:
    if rows is None or cols is None:
        return _make_structured_points_from_unique(points, tolerance)
    else:
        return _make_structured_points_from_corners(points, rows, cols)


def initialize_spline_ctrl_points_perfectly(
        control_points: torch.Tensor,
        world_points: torch.Tensor,
        num_points_x: int,
        num_points_y: int,
        degree_x: int,
        degree_y: int,
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
        change_z_only: bool,
        change_knots: bool,
) -> None:
    new_control_points, new_knots_x, new_knots_y = nurbs.approximate_surface(
        world_points,
        num_points_x,
        num_points_y,
        degree_x,
        degree_y,
        control_points.shape[0],
        control_points.shape[1],
        knots_x if change_knots else None,
        knots_y if change_knots else None,
    )

    if not change_z_only:
        control_points[:, :, :-1] = new_control_points[:, :, :-1]
    control_points[:, :, -1:] = new_control_points[:, :, -1:]
    if change_knots:
        knots_x[:] = new_knots_x
        knots_y[:] = new_knots_y


def initialize_spline_knots_(knots: torch.Tensor, spline_degree: int) -> None:
    num_knot_vals = len(knots[spline_degree:-spline_degree])
    knot_vals = th.linspace(0, 1, num_knot_vals)
    knots[:spline_degree] = 0
    knots[spline_degree:-spline_degree] = knot_vals
    knots[-spline_degree:] = 1


def initialize_spline_knots(
        knots_x: torch.Tensor,
        knots_y: torch.Tensor,
        spline_degree_x: int,
        spline_degree_y: int,
) -> None:
    initialize_spline_knots_(knots_x, spline_degree_x)
    initialize_spline_knots_(knots_y, spline_degree_y)


def calc_ray_diffs(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # We could broadcast here but to avoid a warning, we tile manually.
    # TODO stimmt das so noch?
    return th.nn.functional.l1_loss(pred, target)


def calc_reflection_normals_(
        in_reflections: torch.Tensor,
        out_reflections: torch.Tensor,
) -> torch.Tensor:
    normals = ((in_reflections + out_reflections) / 2 - in_reflections)
    # Handle pass-through "reflection"
    normals = th.where(
        th.isclose(normals, th.zeros_like(normals[0])),
        out_reflections,
        normals / th.linalg.norm(normals, dim=-1).unsqueeze(-1),
    )
    return normals


def calc_reflection_normals(
        in_reflections: torch.Tensor,
        out_reflections: torch.Tensor,
) -> torch.Tensor:
    in_reflections = \
        in_reflections / th.linalg.norm(in_reflections, dim=-1).unsqueeze(-1)
    out_reflections = \
        out_reflections / th.linalg.norm(out_reflections, dim=-1).unsqueeze(-1)
    return calc_reflection_normals_(in_reflections, out_reflections)


def batch_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x * y).sum(-1).unsqueeze(-1)


# def reflect_rays_(rays, normals):
#     return rays - 2 * batch_dot(rays, normals) * normals


# def reflect_rays(rays, normals):
#     normals = normals / th.linalg.norm(normals, dim=-1).unsqueeze(-1)
#     return reflect_rays_(rays, normals)


def save_target(
        heliostat_origin_center: torch.Tensor,
        heliostat_face_normal: torch.Tensor,
        heliostat_points: torch.Tensor,
        heliostat_normals: torch.Tensor,
        heliostat_up_dir: Optional[torch.Tensor],

        receiver_origin_center: torch.Tensor,
        receiver_width: float,
        receiver_height: float,
        receiver_normal: torch.Tensor,
        receiver_up_dir: Optional[torch.Tensor],

        sun: torch.Tensor,
        num_rays: int,
        mean: torch.Tensor,
        cov: torch.Tensor,
        xi: Optional[torch.Tensor],
        yi: Optional[torch.Tensor],

        target_ray_directions: torch.Tensor,
        target_ray_points: torch.Tensor,
        path: str,
) -> None:
    th.save({
        'heliostat_origin_center': heliostat_origin_center,
        'heliostat_face_normal': heliostat_face_normal,
        'heliostat_points': heliostat_points,
        'heliostat_normals': heliostat_normals,
        'heliostat_up_dir': heliostat_up_dir,

        'receiver_origin_center': receiver_origin_center,
        'receiver_width': receiver_width,
        'receiver_height': receiver_height,
        'receiver_normal': receiver_normal,
        'receiver_up_dir': receiver_up_dir,

        'sun': sun,
        'num_rays': num_rays,
        'mean': mean,
        'cov': cov,
        'xi': xi,
        'yi': yi,
    }, path)
