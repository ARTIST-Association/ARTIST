# system dependencies
import torch
import typing
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib
import sys
import os

# local dependencies
module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(module_dir)
from AlignmentModel import AbstractAlignmentModel, alignmentDeviationRadFromNormal, HeliokonAlignmentModel
import AlignmentDisturbanceModel as DM

lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(lib_dir)
import CoordinateSystemsLib.ExtendedCoordinates as COORDS

# Does only work for two axes kinematics
class AlignmentModelAnalyzer:
    result_dir : str = 'AlignmentModelAnalysisResults'
    
    def __init__(self,
                 alignment_model : AbstractAlignmentModel,
                 output_dir_path : typing.Optional[str] = None,
                 dtype: torch.dtype = torch.get_default_dtype(),
                 device: torch.device = torch.device('cpu'),
                ):
        self._dtype = dtype
        self._device = device
        self._alignment_model = alignment_model
        self._output_dir_path = output_dir_path if output_dir_path else os.path.join(lib_dir, AlignmentModelAnalyzer.result_dir)
    
    def fwdbwdAnalysis(self,
                       save_analysis_results : bool = False,
                       show_plot : bool = True,
                       verbose : bool = False,
                       axes_steps : typing.List[int] = [5000, 5000],
                       cmap: str = 'coolwarm'
                        ):
        # alignment position
        alPos = self._alignment_model._position()

        # actuator ranges
        ax1_min = self._alignment_model._actuator_1._min_actuator_steps().item()
        ax1_max = self._alignment_model._actuator_1._max_actuator_steps().item()
        ax2_min = self._alignment_model._actuator_2._min_actuator_steps().item()
        ax2_max = self._alignment_model._actuator_2._max_actuator_steps().item()

        # plotting setup
        fig = plt.figure() if (save_analysis_results or show_plot) else None
        plt_ax1 = fig.add_subplot(1,2,1, projection='3d') if fig else None
        plt_ax2 = fig.add_subplot(1,2,2) if fig else None

        if plt_ax1:
            plt_ax1.set_xlim((alPos[COORDS.E] -1, alPos[COORDS.E] + 1))
            plt_ax1.set_ylim((alPos[COORDS.N] -1, alPos[COORDS.N] + 1))
            plt_ax1.set_zlim((alPos[COORDS.U] -1, alPos[COORDS.U] + 1))
            plt_ax1.set_xlabel('East')
            plt_ax1.set_ylabel('North')
            plt_ax1.set_zlabel('Up')

        if plt_ax2:
            plt_ax2.set_xlabel('Ax1 Steps')
            plt_ax2.set_ylabel('Ax2 Steps')
            plt_ax2.set_xlim((ax1_min, ax1_max))
            plt_ax2.set_ylim((ax2_min, ax2_max))

        # analysis
        ax1, ax2 = torch.meshgrid(torch.linspace(ax1_min, ax1_max, int(ax1_max / axes_steps[0])), torch.linspace(ax2_min, ax2_max, int(ax2_max / axes_steps[1])))
        al_dev = torch.zeros(ax1.shape)
        vmin = None
        vmax = None
        print('STARTING analysis')
        for i, (ax1_row, ax2_row) in enumerate(zip(ax1, ax2)):
            for j, (ax1_elem, ax2_elem) in enumerate(zip(ax1_row, ax2_row)):
                # compute fwd normal
                actuator_steps_fwd = torch.tensor([ax1_elem, ax2_elem])
                normal_fwd, pivoting_point_fwd, side_east_fwd, side_up_fwd, actuator_steps_fwd, cosys_fwd = self._alignment_model.alignmentFromActuatorSteps(actuator_steps=actuator_steps_fwd)
                
                # print fwd results
                np_actuator_steps_fwd = actuator_steps_fwd.detach().numpy()
                np_normal_fwd = normal_fwd.detach().numpy()
                
                # plot fwd results
                np_pivoting_point_fwd = pivoting_point_fwd.detach().numpy()
                if plt_ax1:
                    plt_ax1.plot3D([np_pivoting_point_fwd[COORDS.E], np_pivoting_point_fwd[COORDS.E] + np_normal_fwd[COORDS.E]], 
                              [np_pivoting_point_fwd[COORDS.N], np_pivoting_point_fwd[COORDS.N] + np_normal_fwd[COORDS.N]],
                              [np_pivoting_point_fwd[COORDS.U], np_pivoting_point_fwd[COORDS.U] + np_normal_fwd[COORDS.U]],
                              color='blue'
                              )

                # compute bwd actuator steps
                actuator_steps_bwd = self._alignment_model._actuatorStepsFromNormal(normal=normal_fwd)
                normal_bwd, pivoting_point_bwd, side_east_bwd, side_up_bwd, _, cosys_bwd = self._alignment_model.alignmentFromActuatorSteps(actuator_steps=actuator_steps_bwd)

                # print bwd results
                np_actuator_steps_bwd = actuator_steps_bwd.detach().numpy()
                np_normal_bwd = normal_bwd.detach().numpy()

                # plot bwd results
                np_pivoting_point_bwd = pivoting_point_bwd.detach().numpy()
                if plt_ax1:
                    plt_ax1.plot3D([np_pivoting_point_bwd[COORDS.E], np_pivoting_point_bwd[COORDS.E] + np_normal_bwd[COORDS.E]], 
                              [np_pivoting_point_bwd[COORDS.N], np_pivoting_point_bwd[COORDS.N] + np_normal_bwd[COORDS.N]],
                              [np_pivoting_point_bwd[COORDS.U], np_pivoting_point_bwd[COORDS.U] + np_normal_bwd[COORDS.U]],
                              color='red'
                              )

                # alignment deviation
                al_dev[i][j] = alignmentDeviationRadFromNormal(normal=normal_bwd, normal_target=normal_fwd) * 1000.0
                if not vmax or al_dev[i][j].detach().numpy() > vmax:
                    vmax = al_dev[i][j].detach().numpy()
                if not vmin or al_dev[i][j].detach().numpy() < vmin:
                    vmin = al_dev[i][j].detach().numpy()

                if verbose:
                    print('FWD: Ax1: ' + str(np_actuator_steps_fwd[0]) + ', Ax2: ' + str(np_actuator_steps_fwd[1]) + ' -> ' + str(np_normal_fwd))
                    print('BWD: Ax1: ' + str(np_actuator_steps_bwd[0]) + ', Ax2: ' + str(np_actuator_steps_bwd[1]) + ' -> ' + str(np_normal_bwd))
                    print('DEV: ' + str(al_dev[i][j].detach().numpy()) + 'mRad\n---')

        vmin = vmin if vmin >= 1 else 0
        vmax = vmax if vmax > 1 else 1
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        if plt_ax2:
            plt_ax2.set_title('Alignment Error between FWD and BWD in mRad')
            c = plt_ax2.pcolormesh(ax1.detach().numpy(), ax2.detach().numpy(), al_dev.detach().numpy(), cmap=cmap, norm=norm)
            fig.colorbar(c, ax=plt_ax2)

        if show_plot:
            plt.show()

    def compareModelsByActuatorSteps(self,
                                        cmp_model : AbstractAlignmentModel,
                                        save_analysis_results : bool = False,
                                        show_plot : bool = True,
                                        verbose : bool = False,
                                        axes_steps : typing.List[int] = [1000, 1000],
                                        cmap: str = 'coolwarm',
                        ):
        # alignment position
        alPos = cmp_model._position()

        # actuator ranges
        ax1_min = self._alignment_model._actuator_1._min_actuator_steps().item()
        ax1_max = self._alignment_model._actuator_1._max_actuator_steps().item()
        ax2_min = self._alignment_model._actuator_2._min_actuator_steps().item()
        ax2_max = self._alignment_model._actuator_2._max_actuator_steps().item()

        # plotting setup
        fig = plt.figure() if (save_analysis_results or show_plot) else None
        plt_ax1 = fig.add_subplot(1,2,1, projection='3d') if fig else None
        plt_ax2 = fig.add_subplot(1,2,2) if fig else None
        if plt_ax1:
            plt_ax1.set_xlim((alPos[COORDS.E] -1, alPos[COORDS.E] + 1))
            plt_ax1.set_ylim((alPos[COORDS.N] -1, alPos[COORDS.N] + 1))
            plt_ax1.set_zlim((alPos[COORDS.U] -1, alPos[COORDS.U] + 1))
            plt_ax1.set_xlabel('East')
            plt_ax1.set_ylabel('North')
            plt_ax1.set_zlabel('Up')

        if plt_ax2:
            plt_ax2.set_xlabel('Ax1 Steps')
            plt_ax2.set_ylabel('Ax2 Steps')
            plt_ax2.set_xlim((ax1_min, ax1_max))
            plt_ax2.set_ylim((ax2_min, ax2_max))

        # analysis
        ax1, ax2 = torch.meshgrid(torch.linspace(ax1_min, ax1_max, int(ax1_max / axes_steps[0])), torch.linspace(ax2_min, ax2_max, int(ax2_max / axes_steps[1])))
        al_dev = torch.zeros(ax1.shape)
        print('STARTING analysis')
        for i, (ax1_row, ax2_row) in enumerate(zip(ax1, ax2)):
            for j, (ax1_elem, ax2_elem) in enumerate(zip(ax1_row, ax2_row)):
                # compute fwd normal
                actuator_steps = torch.tensor([ax1_elem, ax2_elem])
                normal_1, pivoting_point_1, side_east_1, side_up_1, actuator_steps_1, cosys_1 = self._alignment_model.alignmentFromActuatorSteps(actuator_steps=actuator_steps)
                normal_2, pivoting_point_2, side_east_2, side_up_2, actuator_steps_2, cosys_2 = cmp_model.alignmentFromActuatorSteps(actuator_steps=actuator_steps)
                
                # print results
                np_actuator_steps_1 = actuator_steps_1.detach().numpy()
                np_normal_1 = normal_1.detach().numpy()
                np_actuator_steps_2 = actuator_steps_2.detach().numpy()
                np_normal_2 = normal_2.detach().numpy()
                
                # plot fwd results
                np_pivoting_point_1 = pivoting_point_1.detach().numpy()
                np_pivoting_point_2 = pivoting_point_2.detach().numpy()
                if plt_ax1:
                    plt_ax1.plot3D([np_pivoting_point_1[COORDS.E], np_pivoting_point_1[COORDS.E] + np_normal_1[COORDS.E]], 
                              [np_pivoting_point_1[COORDS.N], np_pivoting_point_1[COORDS.N] + np_normal_1[COORDS.N]],
                              [np_pivoting_point_1[COORDS.U], np_pivoting_point_1[COORDS.U] + np_normal_1[COORDS.U]],
                              color='red'
                              )
                    plt_ax1.plot3D([np_pivoting_point_2[COORDS.E], np_pivoting_point_2[COORDS.E] + np_normal_2[COORDS.E]], 
                              [np_pivoting_point_2[COORDS.N], np_pivoting_point_2[COORDS.N] + np_normal_2[COORDS.N]],
                              [np_pivoting_point_2[COORDS.U], np_pivoting_point_2[COORDS.U] + np_normal_2[COORDS.U]],
                              color='blue'
                              )

                # alignment deviation
                al_dev[i][j] = alignmentDeviationRadFromNormal(normal=normal_1, normal_target=normal_2) * 1000.0

                if verbose:
                    print('Model 1: Ax1: ' + str(np_actuator_steps_1[0]) + ', Ax2: ' + str(np_actuator_steps_1[1]) + ' -> ' + str(np_normal_1))
                    print('Model 2: Ax1: ' + str(np_actuator_steps_2[0]) + ', Ax2: ' + str(np_actuator_steps_2[1]) + ' -> ' + str(np_normal_2))
                    print('DEV: ' + str(al_dev[i][j].detach().numpy()) + 'mRad\n---')

        if plt_ax2:
            plt_ax2.set_title('Alignment Error between Model 1 and Model 2 in mRad')
            c = plt_ax2.pcolormesh(ax1.detach().numpy(), ax2.detach().numpy(), al_dev.detach().numpy(), cmap=cmap)
            fig.colorbar(c, ax=plt_ax2)

        if show_plot:
            plt.show()

    # only valid for alignment models with disturbance models
    def disturbancesByActuatorSteps(self,
                                        disturbance_list : typing.List[str],
                                        save_analysis_path : typing.Optional[str] = None,
                                        show_plot : bool = True,
                                        verbose : bool = False,
                                        axes_steps : typing.List[int] = [1000, 1000],
                                        value_range: typing.List[float] = [-30, 30],
                                        ncols: int = 5,
                                        cmap: str = 'coolwarm',
                                        env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None,
                                        figsize: typing.Tuple[float, float] = (30,10),
                        ):
        # alignment position
        alPos = self._alignment_model._position()

        # actuator ranges
        ax1_min = self._alignment_model._actuator_1._min_actuator_steps().item()
        ax1_max = self._alignment_model._actuator_1._max_actuator_steps().item()
        ax2_min = self._alignment_model._actuator_2._min_actuator_steps().item()
        ax2_max = self._alignment_model._actuator_2._max_actuator_steps().item()

        # plotting setup
        fig = plt.figure(figsize=figsize) if (save_analysis_path or show_plot) else None
        plt_axes = None
        if fig:
            nrows=int(len(disturbance_list)/ncols + 0.99)
            
            spec = gridspec.GridSpec(ncols=ncols, nrows=nrows,
                                     wspace=0.2, hspace=0.3,
                                    )
            plt_axes = {}
            for key in disturbance_list:
                plt_axes[key] = fig.add_subplot(spec[len(plt_axes)])
                plt_axes[key].set_xlabel('Ax1 Steps')
                plt_axes[key].set_ylabel('Ax2 Steps')
                plt_axes[key].set_xlim((ax1_min, ax1_max))
                plt_axes[key].set_ylim((ax2_min, ax2_max))
                plt_axes[key].set_title(key)
            

        # analysis
        ax1, ax2 = torch.meshgrid(torch.linspace(ax1_min, ax1_max, int(ax1_max / axes_steps[0])), torch.linspace(ax1_min, ax2_max, int(ax2_max / axes_steps[1])))
        al_dev_dict = {}
        for key in disturbance_list:
            al_dev_dict[key] = torch.zeros(ax1.shape)

        vmin = None
        vmax = None

        print('STARTING analysis')
        for i, (ax1_row, ax2_row) in enumerate(zip(ax1, ax2)):
            for j, (ax1_elem, ax2_elem) in enumerate(zip(ax1_row, ax2_row)):
                # compute fwd normal
                actuator_steps = torch.tensor([ax1_elem, ax2_elem])
                disturbance_dict = self._alignment_model.predictDisturbances(actuator_steps=actuator_steps, env_state=env_state)

                for key in disturbance_list:
                    al_dev_dict[key][i][j] = disturbance_dict[key]
                    if not vmin or al_dev_dict[key][i][j].detach().numpy() < vmin:
                        vmin = al_dev_dict[key][i][j].detach().numpy()
                    if not vmax or al_dev_dict[key][i][j].detach().numpy() > vmax:
                        vmax = al_dev_dict[key][i][j].detach().numpy()

                np_actuator_steps = actuator_steps.detach().numpy()
                if verbose:
                    print('Ax1: ' + str(np_actuator_steps[0]) + ', Ax2: ' + str(np_actuator_steps[1]))
                

        if plt_axes and len(plt_axes) > 0:
            for key in disturbance_list:
                # c = plt_axes[key].pcolormesh(ax1.detach().numpy(), ax2.detach().numpy(), al_dev_dict[key].detach().numpy(), cmap=cmap, vmin=vmin, vmax=vmax,)
                c = plt_axes[key].pcolormesh(ax1.detach().numpy(), ax2.detach().numpy(), al_dev_dict[key].detach().numpy(), cmap=cmap)
                fig.colorbar(c, ax=plt_axes[key])

        if show_plot:
            plt.show()

        if save_analysis_path:
            plt.savefig(save_analysis_path, format='png')
            plt.close()

    def toSourceFromAngle(self, az, el):
        el_rad = torch.deg2rad(el)
        az_rad = torch.deg2rad(az - 180)
        e = -torch.sin(az_rad) * torch.cos(el_rad)
        n = -torch.cos(az_rad) * torch.cos(el_rad)
        u = torch.sin(el_rad)
        return torch.tensor([e,n,u], dtype=self._dtype, device=self._device)

    def checkOOB(self,
                 aimpoint : torch.Tensor,
                 angle_steps : typing.List[int] = [5, 5],
                 ) -> torch.Tensor:
        # source orientation range
        min_source_azim = 90
        max_source_azim = 270
        min_source_elev = 0
        max_source_elev = 90

        # actuator ranges
        ax1_min = self._alignment_model._actuator_1._min_actuator_steps().item()
        ax1_max = self._alignment_model._actuator_1._max_actuator_steps().item()
        ax2_min = self._alignment_model._actuator_2._min_actuator_steps().item()
        ax2_max = self._alignment_model._actuator_2._max_actuator_steps().item()
        az, elev = torch.meshgrid(torch.linspace(0, max_source_azim, int(max_source_azim / angle_steps[0]), dtype=self._dtype, device=self._device), torch.linspace(0, max_source_elev, int(max_source_elev / angle_steps[1]), dtype=self._dtype, device=self._device))

        oob_total = torch.tensor(0, dtype=self._dtype, device=self._device)
        for i, (az_row, elev_row) in enumerate(zip(az, elev)):
            for j, (az_elem, elev_elem) in enumerate(zip(az_row, elev_row)):
                to_source = self.toSourceFromAngle(az=az_elem, el=elev_elem)
                normal, pivoting_point, side_east, side_up, actuator_steps, cosys = self._alignment_model.alignmentFromSourceVec(to_source=to_source, aimpoint=aimpoint)
                if actuator_steps[0] < ax1_min:
                    oob_total += torch.abs(actuator_steps[0] - ax1_min) / ax1_max
                if actuator_steps[0] > ax1_max:
                    oob_total += torch.abs(actuator_steps[0] - ax1_max) / ax1_max
                if actuator_steps[1] < ax2_min:
                    oob_total += torch.abs(actuator_steps[1] - ax2_min) / ax2_max
                if actuator_steps[1] > ax2_max:
                    oob_total += torch.abs(actuator_steps[1] - ax2_max) / ax2_max

        return oob_total




    def actuatorStepsBySolarAngles(self,
                                    aimpoint : torch.Tensor,
                                    save_analysis_path : typing.Optional[str] = None,
                                    show_plot : bool = True,
                                    verbose: bool = False,
                                    angle_steps : typing.List[int] = [5, 5],
                                    cmap: str = 'coolwarm',
                                    figsize: typing.Tuple[float, float] = (30,5),
                                    ):
        
        # source orientation range
        min_source_azim = 90
        max_source_azim = 270
        min_source_elev = 0
        max_source_elev = 90
        
        # actuator ranges
        ax1_min = self._alignment_model._actuator_1._min_actuator_steps().item()
        ax1_max = self._alignment_model._actuator_1._max_actuator_steps().item()
        ax2_min = self._alignment_model._actuator_2._min_actuator_steps().item()
        ax2_max = self._alignment_model._actuator_2._max_actuator_steps().item()

        # plotting setup
        norm_ax1 = matplotlib.colors.Normalize(vmin=ax1_min, vmax=ax1_max) 
        norm_ax2 = matplotlib.colors.Normalize(vmin=ax2_min, vmax=ax2_max)
        norm_oor = matplotlib.colors.Normalize(vmin=-1, vmax=1)
        fig = plt.figure(figsize=figsize) if (save_analysis_path or show_plot) else None
        plt_ax1 = fig.add_subplot(1,4,1) if fig else None
        plt_ax2 = fig.add_subplot(1,4,2) if fig else None
        plt_ax3 = fig.add_subplot(1,4,3) if fig else None
        plt_ax4 = fig.add_subplot(1,4,4) if fig else None

        if plt_ax1:
            plt_ax1.set_xlim((min_source_azim, max_source_azim))
            plt_ax1.set_ylim((min_source_elev, max_source_elev))
            plt_ax1.set_xlabel('Source Azimuth')
            plt_ax1.set_ylabel('Source Elevation')

        if plt_ax2:
            plt_ax2.set_xlim((min_source_azim, max_source_azim))
            plt_ax2.set_ylim((min_source_elev, max_source_elev))
            plt_ax2.set_xlabel('Source Azimuth')
            plt_ax2.set_ylabel('Source Elevation')

        if plt_ax3:
            plt_ax3.set_xlim((min_source_azim, max_source_azim))
            plt_ax3.set_ylim((min_source_elev, max_source_elev))
            plt_ax3.set_xlabel('Source Azimuth')
            plt_ax3.set_ylabel('Source Elevation')
        
        if plt_ax4:
            plt_ax4.set_xlim((min_source_azim, max_source_azim))
            plt_ax4.set_ylim((min_source_elev, max_source_elev))
            plt_ax4.set_xlabel('Source Azimuth')
            plt_ax4.set_ylabel('Source Elevation')

        # analysis
        az, elev = torch.meshgrid(torch.linspace(0, max_source_azim, int(max_source_azim / angle_steps[0]), dtype=self._dtype, device=self._device), torch.linspace(0, max_source_elev, int(max_source_elev / angle_steps[1]), dtype=self._dtype, device=self._device))
        ax1 = torch.zeros(az.shape, dtype=self._dtype, device=self._device)
        ax2 = torch.zeros(az.shape, dtype=self._dtype, device=self._device)
        oor1 = torch.zeros(az.shape, dtype=self._dtype, device=self._device)
        oor2 = torch.zeros(az.shape, dtype=self._dtype, device=self._device)

        print('STARTING analysis')
        for i, (az_row, elev_row) in enumerate(zip(az, elev)):
            for j, (az_elem, elev_elem) in enumerate(zip(az_row, elev_row)):
                to_source = self.toSourceFromAngle(az=az_elem, el=elev_elem)
                try:
                    normal, pivoting_point, side_east, side_up, actuator_steps, cosys = self._alignment_model.alignmentFromSourceVec(to_source=to_source, aimpoint=aimpoint)
                    ax1[i,j] = actuator_steps[0]
                    ax2[i,j] = actuator_steps[1]
                except:
                    ax1[i,j] = torch.tensor(ax1_max + 1, dtype=self._dtype, device=self._device)
                    ax2[i,j] = torch.tensor(ax2_max + 1, dtype=self._dtype, device=self._device)

                if ax1[i,j] > ax1_max:
                    oor1[i,j] = 1
                elif ax1[i,j] < ax1_min:
                    oor1[i,j] = -1

                if ax2[i,j] > ax2_max:
                    oor2[i,j] = 2
                elif ax2[i,j] < ax2_min:
                    oor2[i,j] = -1

                if verbose:
                    print(str(az_elem.detach().numpy()) + ', ' + str(elev_elem.detach().numpy()) + ': ' + str(to_source.detach().numpy()) + ' -> ' + str(actuator_steps.detach().numpy()))

        if plt_ax1:
            plt_ax1.set_title('Actuator 1')
            c = plt_ax1.pcolormesh(az.detach().numpy(), elev.detach().numpy(), ax1.detach().numpy(), cmap=cmap, norm=norm_ax1)
            fig.colorbar(c, ax=plt_ax1)

        if plt_ax2:
            plt_ax2.set_title('Actuator 2')
            c = plt_ax2.pcolormesh(az.detach().numpy(), elev.detach().numpy(), ax2.detach().numpy(), cmap=cmap, norm=norm_ax2)
            fig.colorbar(c, ax=plt_ax2)

        if plt_ax3:
            plt_ax3.set_title('Actuator 1: Out Of Range')
            c = plt_ax3.pcolormesh(az.detach().numpy(), elev.detach().numpy(), oor1, cmap=cmap, norm=norm_oor)
            fig.colorbar(c, ax=plt_ax3)

        if plt_ax4:
            plt_ax4.set_title('Actuator 2: Out Of Range')
            c = plt_ax4.pcolormesh(az.detach().numpy(), elev.detach().numpy(), oor2, cmap=cmap, norm=norm_oor)
            fig.colorbar(c, ax=plt_ax4)

        if show_plot:
            plt.show()

        if save_analysis_path:
                plt.savefig(save_analysis_path, format='png')
                plt.close()


def fwdbwdExample():
    torch.set_default_dtype(torch.float64)
    disturbance_list = list(HeliokonAlignmentModel.keys._fields)
    # disturbance_list = [
    #     'position_azim',
    #     'position_elev',
    #     'position_rad',
    #     'joint_1_east_tilt',
    #     'joint_1_north_tilt',
    #     'joint_1_up_tilt',
    #     'joint_2_east_tilt',
    #     'joint_2_north_tilt',
    #     'joint_2_up_tilt',
    #     'actuator_1_increment',
    #     'actuator_2_increment',
    # ]
    disturbance_model = DM.RigidBodyAlignmentDisturbanceModel(disturbance_list=disturbance_list, randomize_initial_disturbances=True, initial_disturbance_range=5.0, dtype=torch.float64)
    # disturbance_model = DM.SNNAlignmentDisturbanceModel(disturbance_list=disturbance_list,
    #                                                  hidden_dim = 2,
    #                                                  n_layers = 3,
    #                                                  num_inputs = 2,
    # )
    aligment_model = HeliokonAlignmentModel(disturbance_model=disturbance_model, max_num_epochs=200, dtype=torch.float64)
    model_analyzer = AlignmentModelAnalyzer(alignment_model=aligment_model, dtype=torch.float64)
    model_analyzer.fwdbwdAnalysis()

def compareByACExample():
    # disturbance_list = list(HeliokonAlignmentModel.keys._fields)
    disturbance_list = [
        'position_azim',
        'position_elev',
        'position_rad',
        'joint_1_east_tilt',
        'joint_1_north_tilt',
        'joint_1_up_tilt',
        'joint_2_east_tilt',
        'joint_2_north_tilt',
        'joint_2_up_tilt',
        'actuator_1_increment',
        'actuator_2_increment',
    ]
    # disturbance_model = DM.RigidBodyAlignmentDisturbanceModel(disturbance_list=disturbance_list, randomize_initial_disturbances=False, initial_disturbance_range=5.0)
    disturbance_model = DM.SNNAlignmentDisturbanceModel(disturbance_list=disturbance_list,
                                                     hidden_dim = 2,
                                                     n_layers = 3,
                                                     num_inputs = 2,
    )
    aligment_model = HeliokonAlignmentModel(disturbance_model=disturbance_model, max_num_epochs=200)
    aligment_model_cmp = HeliokonAlignmentModel(max_num_epochs=200)
    model_analyzer = AlignmentModelAnalyzer(alignment_model=aligment_model)
    model_analyzer.compareModelsByActuatorSteps(cmp_model=aligment_model_cmp)

def distByACExample():
    disturbance_list = list(HeliokonAlignmentModel.keys._fields)
    disturbance_list = disturbance_list[:12]
    disturbance_model = DM.RigidBodyAlignmentDisturbanceModel(disturbance_list=disturbance_list, randomize_initial_disturbances=True, initial_disturbance_range=5.0)
    disturbance_model = DM.SNNAlignmentDisturbanceModel(disturbance_list=disturbance_list,
                                                     hidden_dim = 2,
                                                     n_layers = 3,
                                                     num_inputs = 2,
    )
    aligment_model = HeliokonAlignmentModel(disturbance_model=disturbance_model, max_num_epochs=200)
    model_analyzer = AlignmentModelAnalyzer(alignment_model=aligment_model)
    model_analyzer.disturbancesByActuatorSteps(disturbance_list=disturbance_list)

if __name__ == '__main__':
    # distByACExample()
    compareByACExample()
    # fwdbwdExample()