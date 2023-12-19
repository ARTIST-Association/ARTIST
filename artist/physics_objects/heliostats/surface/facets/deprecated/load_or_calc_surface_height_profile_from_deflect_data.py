
def real_surface(
        real_configs: CfgNode,
        device: torch.device,
) -> HeliostatParams:
    """
    Compute a surface loaded from deflectometric data.
    
    Parameters
    ----------
    real_config : CfgNode
        The config file containing Information about the real surface.
    device : torch.device
        Specifies the device type responsible to load tensors into memory.
    
    Returns
    -------
    HeliostatParams
        Tuple of all heliostat parameters.

    """
    cfg = real_configs
    dtype = torch.get_default_dtype()


    concentratorHeader_struct = struct.Struct(
        cfg.CONCENTRATORHEADER_STRUCT_FMT)
    facetHeader_struct = struct.Struct(cfg.FACETHEADER_STRUCT_FMT)
    ray_struct = struct.Struct(cfg.RAY_STRUCT_FMT)

    (
        surface_position,
        facet_positions,
        facet_spans_n,
        facet_spans_e,
        ideal_positions,
        directions,
        ideal_normal_vecs,
        width,
        height,
    ) = bpro_loader.load_bpro(
        cfg.FILENAME,
        concentratorHeader_struct,
        facetHeader_struct,
        ray_struct,
        cfg.VERBOSE,
    )
    
    surface_position: torch.Tensor = (
        torch.tensor(surface_position, dtype=dtype, device=device)
        if cfg.POSITION_ON_FIELD is None
        else get_position(cfg, dtype, device)
    )

    if cfg.ZS_PATH:
        if cfg.VERBOSE:
            print("Path to surface values found. Load values...")
        positions = copy.deepcopy(ideal_positions)
        integrated = bpro_loader.load_csv(cfg.ZS_PATH, len(positions))
        
        for f_index, facet in enumerate(integrated):
            for p_index, point in enumerate(facet):
                temp = point[0]
                integrated[f_index][p_index][0] = point[1]
                integrated[f_index][p_index][1] = -temp
                integrated[f_index][p_index][2] = point[2]
        

        pos_type = type(positions[0][0][0])

        for (
                facet_index,
                (integrated_facet, pos_facet),
                
        ) in enumerate(zip(integrated, positions)):

            integrated_facet_iter = iter(integrated_facet)
            in_facet_index = 0
            while in_facet_index < len(pos_facet):
                curr_integrated = next(integrated_facet_iter)
                pos = pos_facet[in_facet_index]

                # Remove positions without matching integrated.
                rounded_pos = [round(val, 4) for val in pos[:-1]]
                rounded_integrated = [
                    round(val, 4)
                    for val in curr_integrated[:-1]
                ]
                while not all(map(
                        lambda tup: tup[0] == tup[1],
                        zip(rounded_pos, rounded_integrated),
                )):
                    pos_facet.pop(in_facet_index)
                    directions[facet_index].pop(in_facet_index)
                    ideal_normal_vecs[facet_index].pop(in_facet_index)
                    if in_facet_index >= len(pos_facet):
                        break

                    pos = pos_facet[in_facet_index]
                    rounded_pos = [round(val, 4) for val in pos[:-1]]
                else:
                    pos = list(pos)
                    pos[-1] = pos_type(curr_integrated[-1])
                    in_facet_index += 1
        del integrated
    else:
        positions = ideal_positions

    h_normal_vecs = []
    h_ideal_vecs = []
    h = []
    h_ideal = []
    if not cfg.ZS_PATH:
        if cfg.VERBOSE:
            print(
                "No path to surface surface values found. "
                "Calculate values..."
            )
        zs_list = []
    step_size = sum(map(len, directions)) // cfg.TAKE_N_VECTORS
    for f in range(len(directions)):
        h_normal_vecs.append(torch.tensor(
            directions[f][::step_size],
            dtype=dtype,
            device=device,
        ))
        h_ideal_vecs.append(torch.tensor(
            ideal_normal_vecs[f][::step_size],
            dtype=dtype,
            device=device,
        ))
        h.append(torch.tensor(
            ideal_positions[f][::step_size],
            dtype=dtype,
            device=device,
        ))
        if not cfg.ZS_PATH:
            zs_list.append(deflec_facet_zs_many(
                h[-1],
                h_normal_vecs[-1],
                h_ideal_vecs[-1],
                num_samples=16,
            ))
        h_ideal.append(torch.tensor(
            ideal_positions[f][::step_size],
            dtype=dtype,
            device=device,
        ))

    h_normal_vecs: torch.Tensor = torch.cat(h_normal_vecs, dim=0)
    h_ideal_vecs: torch.Tensor = torch.cat(h_ideal_vecs, dim=0)
    h: torch.Tensor = torch.cat(h, dim=0)
    if not cfg.ZS_PATH:
        zs = torch.cat(zs_list, dim=0)
        h[:, -1] += zs

    h_ideal: torch.Tensor = torch.cat(h_ideal, dim=0)
    if cfg.VERBOSE:
        print("Done")
    
    ###################################################################
    # fig = plt.figure(figsize =(14, 9))
    # ax = plt.axes(projection ='3d')
    # h[:,2] = h[:,2]-h_ideal[:,2]
    # h = h.detach().cpu()
    # # im3 = ax.scatter(h[:,0],h[:,1], c=h[:,2], cmap="magma")
    # # h.detach().cpus()
    # my_cmap = plt.get_cmap('hot')
    # ax.plot_trisurf(h[:,0],h[:,1],h[:,2], cmap =my_cmap)
    # plt.show()
    ####################################################################
        
    rows = None
    cols = None
    params = None
    return (
        surface_position,
        torch.tensor(facet_positions, dtype=dtype, device=device),
        torch.tensor(facet_spans_n, dtype=dtype, device=device),
        torch.tensor(facet_spans_e, dtype=dtype, device=device),
        h,
        h_ideal,
        h_normal_vecs,
        h_ideal_vecs,
        height,
        width,
        rows,
        cols,
        params,
    )


def deflec_facet_zs_many(
        points: torch.Tensor,
        normals: torch.Tensor,
        normals_ideal: torch.Tensor,
        num_samples: int = 4,
        use_weighted_average: bool = False,
        eps: float = 1e-6,
) -> torch.Tensor:
    """
    Calculate z values for a surface given by normals at x-y-planar
    positions.

    We are left with a different unknown offset for each z value; we
    assume this to be constant.
    
    Parameters
    ----------
    points : torch.Tensor
        Points on the given surface.
    normals : torch.Tensor 
        Normal vectors coressponding to the points.
    normals_ideal : torch.Tensor
        Ideal normal vectors.
    num_samples : int = 4
        Number of samples.
    use_weighted_average : bool = False
        Wether or not to use weighted averages.
    eps : float = 1e-6
        Cut-off value/ limit.
    
    Returns
    -------
    torch.Tensor
        The z values.
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
        angle_indices = torch.argmax(angles_in_slice.long(), dim=-1)
    else:
        angle_indices = torch.empty(
            (0, len(points)), dtype=torch.long, device=device)

    # Select the angles we found for each slice.
    angles = torch.gather(angles.squeeze(0), -1, angle_indices.T)

    # Handle _not_ having found an angle. We here create an array of
    # booleans, indicating whether we found an angle, for each slice.
    found_angles = torch.gather(
        angles_in_slice,
        -1,
        angle_indices.unsqueeze(-1),
    ).squeeze(-1)
    # We always found something in the zeroth slice, so add those here.
    found_angles = torch.cat([
        torch.ones((1,) + found_angles.shape[1:], dtype=torch.bool, device=device),
        found_angles,
    ], dim=0)
    del angles_in_slice

    # Set up some numbers for averaging.
    if use_weighted_average:
        angle_diffs = (
            torch.cat([
                torch.zeros((len(angles), 1), dtype=dtype, device=device),
                angles,
            ], dim=-1)
            - angle_slices.squeeze(-1).T
        )
        # Inverse difference in angle.
        weights = 1 / (angle_diffs + eps).T
        del angle_diffs
    else:
        # Number of samples we found angles for.
        num_available_samples = torch.count_nonzero(found_angles, dim=0)

    # Finally, combine the indices of the closest points (zeroth slice)
    # with the indices of all closest points in the other slices.
    closest_indices = torch.cat((
        closest_indices.unsqueeze(0),
        angle_indices,
    ), dim=0)
    del angle_indices, angle_slices

    midway_normal = normals + normals[closest_indices]
    midway_normal /= torch.linalg.norm(midway_normal, dim=-1, keepdims=True)

    rot_90deg = axis_angle_rotation(
        normals_ideal, torch.tensor(math.pi / 2, dtype=dtype, device=device))

    connector = points[closest_indices] - points
    connector_norm = torch.linalg.norm(connector, dim=-1)
    orthogonal = torch.matmul(
        rot_90deg.unsqueeze(0),
        connector.unsqueeze(-1),
    ).squeeze(-1)
    orthogonal /= torch.linalg.norm(orthogonal, dim=-1, keepdims=True)
    tilted_connector = torch.cross(orthogonal, midway_normal, dim=-1)
    tilted_connector /= torch.linalg.norm(tilted_connector, dim=-1, keepdims=True)
    tilted_connector *= torch.sign(connector[..., -1]).unsqueeze(-1)

    angle = torch.acos(torch.clamp(
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
    angle = torch.where(
        found_angles & ~torch.isnan(angle),
        angle,
        torch.tensor(0.0, dtype=dtype, device=device),
    )

    # Average over each slice.
    if use_weighted_average:
        zs = (
            (weights * connector_norm * torch.tan(angle)).sum(dim=0)
            / (weights * found_angles.to(dtype)).sum(dim=0)
        )
    else:
        zs = (
            (connector_norm * torch.tan(angle)).sum(dim=0)
            / num_available_samples
        )

    return zs

def horizontal_distance(
        a: torch.Tensor,
        b: torch.Tensor,
        ord: Union[int, float, str] = 2,
) -> torch.Tensor:
    """Return the horizontal distance between a and b"""
    return torch.linalg.norm(b[..., :-1] - a[..., :-1], dim=-1, ord=ord)

def _all_angles(
        points: torch.Tensor,
        normals: torch.Tensor,
        closest_indices: torch.Tensor,
        remaining_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate angles between direction to closest point and all others

    Parameters
    ----------
    points : torch.Tensor
        The points to be considered.
    normals : torch.Tensor
        The normals corresponding to the points.
    closest_indices : torch.Tensor
        Indices of the closest points.
    remaining_indices : torch.Tensor
        Indices of the remaining points.
    
    Returns
    -------
    torch.Tensor
        The angles between the direction of the closest point and all others.
    """
    connector = (points[closest_indices] - points).unsqueeze(1)
    other_connectors = (
        points[remaining_indices]
        - points.unsqueeze(1)
    )
    angles = torch.acos(torch.clamp(
        (
            batch_dot(connector, other_connectors).squeeze(-1)
            / (
                torch.linalg.norm(connector, dim=-1)
                * torch.linalg.norm(other_connectors, dim=-1)
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
                torch.cross(
                    torch.tile(connector, (1, other_connectors.shape[1], 1)),
                    other_connectors,
                    dim=-1,
                ),
            ).squeeze(-1)
            < 0
        )
    )

    # And convert to 360° rotations.
    tau = 2 * torch.tensor(math.pi, dtype=angles.dtype, device=angles.device)
    angles = torch.where(angles < 0, tau + angles, angles)
    return angles



def _find_angles_in_other_slices(
        angles: torch.Tensor,
        num_slices: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find positions of all angles in each slice

    Parameters
    ----------
    angles : torch.Tensor
        The angles.
    num_slices : int 
        The number of slices.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The positions of all angles ine ach slice and all the slices.
    """
    dtype = angles.dtype
    device = angles.device
    # Set up uniformly sized cake/pizza slices for which to find angles.
    tau = 2 * torch.tensor(math.pi, dtype=dtype, device=device)
    angle_slice = tau / num_slices

    angle_slices = (
        torch.arange(
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

