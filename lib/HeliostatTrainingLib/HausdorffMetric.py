# system dependencies
import torch
import typing
# import sys
# import os
# local dependencies
# module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
# sys.path.append(module_dir)
# lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
# sys.path.append(lib_dir)

class HausdorffMetric:

    def __init__(self):
        pass

    def distanceByAngle(data_point_from,
                 data_point_to,
                ):
        el_dist = (data_point_from.sourceElev() - data_point_to.sourceElev()) ** 2
        az_dist = (data_point_from.sourceAzim() - data_point_to.sourceAzim()) ** 2
        return torch.sqrt(el_dist + az_dist)
    
    def distanceByAngle_angles(azim, elev, data_point_to):
        el_dist = (elev - data_point_to.sourceElev()) ** 2
        az_dist = (azim - data_point_to.sourceAzim()) ** 2
        return torch.sqrt(el_dist + az_dist)

    def distanceBySteps(data_point_from,
                 data_point_to,
                ):
        ax1_dist = (data_point_from.ax1_steps - data_point_to.ax1_steps) ** 2
        ax2_dist = (data_point_from.ax2_steps - data_point_to.ax2_steps) ** 2
        # el_dist = (data_point_from.sourceElev() - data_point_to.sourceElev()) ** 2
        # az_dist = (data_point_from.sourceAzim() - data_point_to.sourceAzim()) ** 2
        return torch.sqrt(ax1_dist + ax2_dist)

    def distanceToDataset(data_point,
                          data_points : typing.Dict[int, any],
                          selected_data_points : typing.List[typing.List[int]] = [],
                          dist_method : typing.Callable = distanceByAngle,
                          num_nearest_neighbors : int = 1,
                            ):
        min_dist_arr = []
        neigh_dict = {}
        selected_data_points = selected_data_points if selected_data_points else [list(data_points.keys())]

        for dataset in selected_data_points:
            if len(dataset) > 0:
                for key in dataset:
                    if (not key in data_points.keys() or data_point.id == key):
                        continue

                    h_dist = dist_method(data_point_from=data_point, data_point_to=data_points[key])
                    
                    if len(min_dist_arr) < num_nearest_neighbors:
                        min_dist_arr.append(h_dist)
                        min_dist_arr.sort(reverse=True) # -> [10, 9, 8, 7, ...]
                        neigh_dict[h_dist] = key
                    else:
                        if h_dist < min_dist_arr[0]:
                            min_dist_arr[0] = h_dist
                            min_dist_arr.sort(reverse=True) # -> [10, 9, 8, 7, ...]
                            neigh_dict[h_dist] = key
                                
        min_dist = sum(min_dist_arr)
        neigh_arr = [neigh_dict[d] for d in min_dist_arr]

        return min_dist, neigh_arr
    
    def distanceToDataset_angles(azim,
                                 elev,
                                data_points : typing.Dict[int, any],
                                dist_method : typing.Callable = distanceByAngle_angles,
                                num_nearest_neighbors : int = 1,
                                    ):
        min_dist_arr = []

        for dp in data_points.values():

            h_dist = dist_method(azim=azim, elev=elev, data_point_to=dp)
            
            if len(min_dist_arr) < num_nearest_neighbors:
                min_dist_arr.append(h_dist)
                min_dist_arr.sort(reverse=True) # -> [10, 9, 8, 7, ...]
            else:
                if h_dist < min_dist_arr[0]:
                    min_dist_arr[0] = h_dist
                    min_dist_arr.sort(reverse=True) # -> [10, 9, 8, 7, ...]
                                
        min_dist = sum(min_dist_arr)

        return min_dist