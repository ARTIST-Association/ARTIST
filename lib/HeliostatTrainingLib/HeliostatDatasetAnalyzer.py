# system dependencies
import torch
import typing
import datetime
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import sys
import os

# local dependencies
module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(module_dir)
from HeliostatDatapoint import HeliostatDataPoint
from HeliostatDataset import HeliostatDataset
from HausdorffMetric import HausdorffMetric
# lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
# sys.path.append(lib_dir)

def gauss(x, mu, sig):
    # return (1 / np.sqrt(2 * np.pi * (sig**2))) * np.exp((-1/2) * ((x-mu) / sig)**2)
    return (1 / torch.sqrt(2 * torch.pi * (sig**2))) * torch.exp((-1/2) * (x-mu)**2 / (sig**2))

class HeliostatDatasetAnalyzer:

    def __init__(self,
                 dataset: HeliostatDataset,
                 dtype: torch.dtype = torch.get_default_dtype(),
                 device: torch.device = torch.device('cpu')
                ):
        self._dataset = dataset
        self._dtype = dtype
        self._device = device

    def plotHausdorffGauss(self,
                           show_plot : bool = True,
                           save_path : typing.Optional[str] = None,
                           angle_steps : typing.List[int] = [2, 2],
                          ):
        [
            all_ax1, all_ax2, all_azim, all_elev, all_dates,
            train_ax1, train_ax2, train_azim, train_elev, train_dates,
            test_ax1, test_ax2, test_azim, test_elev, test_dates,
            eval_ax1, eval_ax2, eval_azim, eval_elev, eval_dates,
            train_err, test_err, eval_err, train_hausdorff, test_hausdorff, eval_hausdorff, vmin, vmax, vmin_hausdorff, vmax_hausdorff, max_ax
        ] = self._dataset.prepareDataForPlotting(plot_errors=False, plot_hausdorff=True)

        # source orientation range
        min_source_azim = 60
        max_source_azim = 300
        min_source_elev = 0
        max_source_elev = 90
        
        az, elev = torch.meshgrid(torch.linspace(min_source_azim, max_source_azim, int((max_source_azim - min_source_azim) / angle_steps[0]), dtype=self._dtype, device=self._device), torch.linspace(min_source_elev, max_source_elev, int((max_source_elev - min_source_elev) / angle_steps[1]), dtype=self._dtype, device=self._device))
        hd_list = torch.zeros(az.shape, dtype=self._dtype, device=self._device)
        for i, (az_row, elev_row) in enumerate(zip(az, elev)):
            for j, (az_elem, elev_elem) in enumerate(zip(az_row, elev_row)):
                to_source = self.toSourceFromAngle(az=az_elem, el=elev_elem)

                # data point dummy
                dp = HeliostatDataPoint(id = 0,
                                        heliostat='', 
                                        ax1_steps=torch.tensor(0, dtype=self._dtype, device=self._device),
                                        ax2_steps=torch.tensor(0, dtype=self._dtype, device=self._device),
                                        aimpoint=torch.tensor([0,0,0], dtype=self._dtype, device=self._device),
                                        created_at=datetime.datetime.now(),
                                        to_source=to_source,
                                        )
                hd_list[i][j] = dp.distanceToDataset(data_points=self._dataset._data_points,
                                                       selected_data_points=[list(self._dataset.trainingDataset().keys())],
                                                        )

        hd_list, _ = torch.sort(torch.flatten(hd_list))

        hd_mean = torch.mean(hd_list)
        hd_std = torch.std(hd_list)

        if show_plot or save_path:
            fig = plt.figure(figsize=(30,10))
            plt_ax1 = fig.add_subplot(1,1,1)

            hd_gauss = gauss(x=hd_list, mu=hd_mean, sig=hd_std)
            plt_ax1.fill_between(hd_list.detach().numpy(), hd_gauss.detach().numpy())
            plt_ax1.plot([hd_mean, hd_mean], [0, 2.0], c='orange')
            plt_ax1.set_xlabel('Hausdorff Distance [Rad]')
            plt_ax1.set_ylabel('Frequency [%]')
            hd_list_max = torch.max(hd_list).item() * 1.05
            gauss_max = torch.max(hd_gauss).item() * 1.05
            plt_ax1.set_xlim((0, 2.0))
            plt_ax1.set_ylim((0, 2.0))

            if show_plot:
                plt.show()

            if save_path:
                plt.savefig(save_path, format='png')
                plt.close()

        return hd_mean, hd_std

    def plotDataDistributionOverAxes(self, 
                                     plot_errors : bool = False,
                                     plot_hausdorff : bool = False,
                                     plot_epsilon_regions : bool = False,
                                     show_plot : bool = True,
                                     cmap_str : str = 'magma',
                                     save_path : typing.Optional[str] = None,
                                     figsize: typing.Tuple[float, float] = (30,20),
                                     ):
        matplotlib.rcParams.update({'font.size': 22})
        
        [
            all_ax1, all_ax2, all_azim, all_elev, all_dates,
            train_ax1, train_ax2, train_azim, train_elev, train_dates,
            test_ax1, test_ax2, test_azim, test_elev, test_dates,
            eval_ax1, eval_ax2, eval_azim, eval_elev, eval_dates,
            train_err, test_err, eval_err, train_hausdorff, test_hausdorff, eval_hausdorff, vmin, vmax, vmin_hausdorff, vmax_hausdorff, max_ax
        ] = self._dataset.prepareDataForPlotting(plot_errors=plot_errors, plot_hausdorff=plot_hausdorff)

        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        if plot_errors:
            cmap = matplotlib.cm.get_cmap(cmap_str)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            if len(all_ax1) > 0:
                ax.scatter(all_ax2, all_ax1, c='grey', alpha=0.2, label='other')
            # if len(initial_ax1) > 0:
            #     ax.scatter(initial_ax2, initial_ax1, c=cmap(norm(initial_err)), label='initial_err')
            if len(train_ax1) > 0:
                ax.scatter(train_ax2, train_ax1, c=cmap(norm(train_err)), label='train', marker='s')
            if len(test_ax1) > 0:
                ax.scatter(test_ax2, test_ax1, c=cmap(norm(test_err)), label='test', marker='D')
            if len(eval_ax1) > 0:
                ax.scatter(eval_ax2, eval_ax1, c=cmap(norm(eval_err)), label='eval', marker='*', s=150, edgecolors='black')
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        elif plot_hausdorff:
            cmap = matplotlib.cm.get_cmap(cmap_str)
            norm = matplotlib.colors.Normalize(vmin=vmin_hausdorff, vmax=vmax_hausdorff)
            # if len(all_ax1) > 0:
            #     ax.scatter(all_ax2, all_ax1, c='grey', alpha=0.2, label='other')
            # if len(initial_ax1) > 0:
            #     ax.scatter(initial_ax2, initial_ax1, c=cmap(norm(initial_err)), label='initial_err')
            if len(train_ax1) > 0:
                ax.scatter(train_ax2, train_ax1, c=cmap(norm(train_hausdorff)), label='train', marker='s', alpha = 0.2)
            if len(test_ax1) > 0:
                if plot_epsilon_regions:
                    for ax1, ax2, hd in zip(test_ax1, test_ax2, test_hausdorff):
                        circle = plt.Circle((ax2, ax1), hd, color='green', fill=False, clip_on=True, alpha=0.2)
                        ax.add_patch(circle)
                ax.scatter(test_ax2, test_ax1, c=cmap(norm(test_hausdorff)), label='test', marker='D')
            if len(eval_ax1) > 0:
                if plot_epsilon_regions:
                    for ax1, ax2, hd in zip(eval_ax1, eval_ax2, eval_hausdorff):
                        circle = plt.Circle((ax2, ax1), hd, color='orange', fill=False, clip_on=True, alpha=0.4)
                        ax.add_patch(circle)
                ax.scatter(eval_ax2, eval_ax1, c=cmap(norm(eval_hausdorff)), label='eval', marker='*', s=150, edgecolors='black')
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        else:
            if len(all_ax1) > 0:
                ax.scatter(all_ax2, all_ax1, c='grey', alpha=0.2, label='other')
            if len(train_ax1) > 0:
                ax.scatter(train_ax2, train_ax1, c='blue', label='train')
            if len(test_ax1) > 0:
                ax.scatter(test_ax2, test_ax1, c='green', label='test')
            if len(eval_ax1) > 0:
                ax.scatter(eval_ax2, eval_ax1, c='orange', label='eval')
        if plot_errors:
            ax.set_title('Alignment Error [Rad]')
        if plot_hausdorff:
            ax.set_title('Hausdorff Distance [Rad]')
        ax.set_ylabel('Ax1 Steps')
        ax.set_xlabel('Ax2 Steps')
        ax.set_xlim((0,max_ax*1.05))
        ax.set_ylim((0,max_ax*1.05))
        ax.legend()

        if show_plot:
            plt.show()

        if save_path:
            plt.savefig(save_path, format='png')
            plt.close()

    def plotDataDistributionOverAngles(self, 
                                     plot_errors : bool = False,
                                     plot_hausdorff : bool = False,
                                     plot_epsilon_regions : bool = False,
                                     show_plot : bool = True,
                                     cmap_str : str = 'magma',
                                     save_path : typing.Optional[str] = None,
                                     figsize: typing.Tuple[float, float] = (30,8),
                                     ):
        matplotlib.rcParams.update({'font.size': 22})
        plot_epsilon_regions = True
        
        [
            all_ax1, all_ax2, all_azim, all_elev, all_dates,
            train_ax1, train_ax2, train_azim, train_elev, train_dates,
            test_ax1, test_ax2, test_azim, test_elev, test_dates,
            eval_ax1, eval_ax2, eval_azim, eval_elev, eval_dates,
            train_err, test_err, eval_err, train_hausdorff, test_hausdorff, eval_hausdorff, vmin, vmax, vmin_hausdorff, vmax_hausdorff, max_ax
        ] = self._dataset.prepareDataForPlotting(plot_errors=plot_errors, plot_hausdorff=plot_hausdorff)

        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        if plot_errors:
            cmap = matplotlib.cm.get_cmap(cmap_str)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            if len(all_azim) > 0:
                ax.scatter(all_azim, all_elev, c='grey', alpha=0.2, label='other')
            if len(train_azim) > 0:
                ax.scatter(train_azim, train_elev, c=cmap(norm(train_err)), label='train', marker='s')
            if len(test_azim) > 0:
                ax.scatter(test_azim, test_elev, c=cmap(norm(test_err)), label='test', marker='D')
            if len(eval_azim) > 0:
                ax.scatter(eval_azim, eval_elev, c=cmap(norm(eval_err)), label='eval', marker='*', s=150, edgecolors='black')
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        elif plot_hausdorff:
            cmap = matplotlib.cm.get_cmap(cmap_str)
            norm = matplotlib.colors.Normalize(vmin=vmin_hausdorff, vmax=vmax_hausdorff)
            if len(train_azim) > 0:
                ax.scatter(train_azim, train_elev, c=cmap(norm(train_hausdorff)), label='train', marker='s', alpha = 0.2)
            if len(test_azim) > 0:
                if plot_epsilon_regions:
                    for azim, elev, hd in zip(test_azim, test_elev, test_hausdorff):
                        circle = plt.Circle((azim, elev), hd / 3.14 * 180 , color='green', fill=False, clip_on=True, alpha=0.2)
                        ax.add_patch(circle)
                ax.scatter(test_azim, test_elev, c=cmap(norm(test_hausdorff)), label='test', marker='D')
            if len(eval_azim) > 0:
                if plot_epsilon_regions:
                    for azim, elev, hd in zip(eval_azim, eval_elev, eval_hausdorff):
                        circle = plt.Circle((azim, elev), hd / 3.14 * 180, color='orange', fill=False, clip_on=True, alpha=0.2)
                        ax.add_patch(circle)
                ax.scatter(eval_azim, eval_elev, c=cmap(norm(eval_hausdorff)), label='eval', marker='*', s=150, edgecolors='black')

            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        else:
            if len(all_azim) > 0:
                ax.scatter(all_azim, all_elev, c='grey', alpha=0.2, label='other')
            if len(train_azim) > 0:
                ax.scatter(train_azim, train_elev, c='blue', label='train')
            if len(test_azim) > 0:
                ax.scatter(test_azim, test_elev, c='green', label='test')
            if len(eval_azim) > 0:
                ax.scatter(eval_azim, eval_elev, c='orange', label='eval')
        if plot_errors:
            ax.set_title('Alignment Error [Rad]')
        if plot_hausdorff:
            ax.set_title('Hausdorff Distance [Rad]')
        ax.set_ylabel('Source Elevation [Deg]')
        ax.set_xlabel('Source Azimuth [Deg]')
        ax.set_xlim((70,300))
        ax.set_ylim((0,90))
        ax.legend()

        if show_plot:
            plt.show()

        if save_path:
            plt.savefig(save_path, format='png')
            plt.close()

    def plotDataDistributionOverDates(self, 
                                     plot_errors : bool = False,
                                     plot_hausdorff : bool = False,
                                     show_plot : bool = True,
                                     cmap_str : str = 'magma',
                                     save_path : typing.Optional[str] = None,
                                     figsize: typing.Tuple[float, float] = (40,20),
                                     ):
        matplotlib.rcParams.update({'font.size': 22})
        
        [
            all_ax1, all_ax2, all_azim, all_elev, all_dates,
            train_ax1, train_ax2, train_azim, train_elev, train_dates,
            test_ax1, test_ax2, test_azim, test_elev, test_dates,
            eval_ax1, eval_ax2, eval_azim, eval_elev, eval_dates,
            train_err, test_err, eval_err, train_hausdorff, test_hausdorff, eval_hausdorff, vmin, vmax, vmin_hausdorff, vmax_hausdorff, max_ax
        ] = self._dataset.prepareDataForPlotting(plot_errors=plot_errors, plot_hausdorff=plot_hausdorff)

        all_times = [d.hour + d.minute / 60 for d in all_dates]
        train_times = [d.hour + d.minute / 60 for d in train_dates]
        test_times = [d.hour + d.minute / 60 for d in test_dates]
        eval_times = [d.hour + d.minute / 60 for d in eval_dates]

        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        if plot_errors:
            cmap = matplotlib.cm.get_cmap(cmap_str)
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            if len(all_ax1) > 0:
                ax.scatter(all_dates, all_times, c='grey', alpha=0.2, label='other')
            # if len(initial_ax1) > 0:
            #     ax.scatter(initial_ax2, initial_ax1, c=cmap(norm(initial_err)), label='initial_err')
            if len(train_ax1) > 0:
                ax.scatter(train_dates, train_times, c=cmap(norm(train_err)), label='train', marker='s')
            if len(test_ax1) > 0:
                ax.scatter(test_dates, test_times, c=cmap(norm(test_err)), label='test', marker='D')
            if len(eval_ax1) > 0:
                ax.scatter(eval_dates, eval_times, c=cmap(norm(eval_err)), label='eval', marker='*', s=150, edgecolors='black')
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        elif plot_hausdorff:
            cmap = matplotlib.cm.get_cmap(cmap_str)
            norm = matplotlib.colors.Normalize(vmin=vmin_hausdorff, vmax=vmax_hausdorff)
            # if len(all_ax1) > 0:
            #     ax.scatter(all_ax2, all_ax1, c='grey', alpha=0.2, label='other')
            # if len(initial_ax1) > 0:
            #     ax.scatter(initial_ax2, initial_ax1, c=cmap(norm(initial_err)), label='initial_err')
            if len(train_ax1) > 0:
                ax.scatter(train_dates, train_times, c=cmap(norm(train_hausdorff)), label='train', marker='s', alpha = 0.2)
            if len(test_ax1) > 0:
                ax.scatter(test_dates, test_times, c=cmap(norm(test_hausdorff)), label='test', marker='D')
            if len(eval_ax1) > 0:
                ax.scatter(eval_dates, eval_times, c=cmap(norm(eval_hausdorff)), label='eval', marker='*', s=150, edgecolors='black')
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        else:
            if len(all_ax1) > 0:
                ax.scatter(all_dates, all_times, c='grey', alpha=0.2, label='other')
            if len(train_ax1) > 0:
                ax.scatter(train_dates, train_times, c='blue', label='train')
            if len(test_ax1) > 0:
                ax.scatter(test_dates, test_times, c='green', label='test')
            if len(eval_ax1) > 0:
                ax.scatter(eval_dates, eval_times, c='orange', label='eval')
        if plot_errors:
            ax.set_title('Alignment Error [Rad]')
        if plot_hausdorff:
            ax.set_title('Hausdorff Distance [Rad]')
        ax.set_ylabel('Hour')
        ax.set_xlabel('Date')
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=30))
        # ax.set_ylim((0,max_ax*1.05))
        # ax.yaxis.set_major_locator(matplotlib.dates.HourLocator(interval=1))
        # ax.yaxis.set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        # ax.set_yticks([])
        ax.legend()
        fig.autofmt_xdate()
        
        if show_plot:
            plt.show()

        if save_path:
            plt.savefig(save_path, format='png')
            plt.close()

    def toSourceFromAngle(self, az, el):
        el_rad = torch.deg2rad(el)
        az_rad = torch.deg2rad(az - 180)
        e = -torch.sin(az_rad) * torch.cos(el_rad)
        n = -torch.cos(az_rad) * torch.cos(el_rad)
        u = torch.sin(el_rad)
        return torch.tensor([e,n,u], dtype=self._dtype, device=self._device)

    def plotDataDistributionOverHausdorff(self,
                                        show_plot : bool = True,
                                        cmap_str : str = 'magma',
                                        save_path : typing.Optional[str] = None,
                                        figsize: typing.Tuple[float, float] = (40,20),
                                        angle_steps : typing.List[int] = [2, 2],
                                        ):
        matplotlib.rcParams.update({'font.size': 22})

        [
            all_ax1, all_ax2, all_azim, all_elev, all_dates,
            train_ax1, train_ax2, train_azim, train_elev, train_dates,
            test_ax1, test_ax2, test_azim, test_elev, test_dates,
            eval_ax1, eval_ax2, eval_azim, eval_elev, eval_dates,
            train_err, test_err, eval_err, train_hausdorff, test_hausdorff, eval_hausdorff, vmin, vmax, vmin_hausdorff, vmax_hausdorff, max_ax
        ] = self._dataset.prepareDataForPlotting(plot_errors=True, plot_hausdorff=True)

        reg_params = self._dataset.computeRegressionParameters()

        avg_train = self._dataset.avgTrainError().item()
        avg_test = self._dataset.avgTestError().item()
        avg_eval = self._dataset.avgEvalError().item()

        vmax_hausdorff = vmax_hausdorff*1.05

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        # source orientation range
        min_source_azim = 60
        max_source_azim = 300
        min_source_elev = 0
        max_source_elev = 90

        hd_range = torch.linspace(0, vmax_hausdorff, 2)
        hd_ones = torch.ones(hd_range.size()).cpu().detach().numpy()
        err_est = reg_params[1]*hd_range+reg_params[0]
        hd_range = hd_range.cpu().detach().numpy()
        err_est = err_est.cpu().detach().numpy()
        if avg_train:
            ax.plot(hd_range, hd_ones * avg_train, alpha=0.2, c='blue')
        if avg_test:
            ax.plot(hd_range, hd_ones * avg_test, alpha=0.2, c='green')
        if avg_eval:
            ax.plot(hd_range, hd_ones * avg_eval, alpha=0.2, c='orange')

        ax.plot(hd_range, err_est, alpha = 0.2, c='grey')

        if len(train_ax1) > 0:
            ax.scatter(train_hausdorff, train_err, c='blue', alpha=0.2, label='train', marker='s')
        if len(test_ax1) > 0:
            ax.scatter(test_hausdorff, test_err, c='green', alpha=0.2, label='test', marker='D')
        if len(eval_ax1) > 0:
            ax.scatter(eval_hausdorff, eval_err, c='orange', label='eval', marker='*', s=150, edgecolors='black')

        az, elev = torch.meshgrid(torch.linspace(min_source_azim, max_source_azim, int((max_source_azim - min_source_azim) / angle_steps[0]), dtype=self._dtype, device=self._device), torch.linspace(min_source_elev, max_source_elev, int((max_source_elev - min_source_elev) / angle_steps[1]), dtype=self._dtype, device=self._device))
        err_est_mesh = torch.zeros(az.shape, dtype=self._dtype, device=self._device)
        for i, (az_row, elev_row) in enumerate(zip(az, elev)):
            for j, (az_elem, elev_elem) in enumerate(zip(az_row, elev_row)):
                to_source = self.toSourceFromAngle(az=az_elem, el=elev_elem)

                # data point dummy
                dp = HeliostatDataPoint(id = 0,
                                        heliostat='', 
                                        ax1_steps=torch.tensor(0, dtype=self._dtype, device=self._device),
                                        ax2_steps=torch.tensor(0, dtype=self._dtype, device=self._device),
                                        aimpoint=torch.tensor([0,0,0], dtype=self._dtype, device=self._device),
                                        created_at=datetime.datetime.now(),
                                        to_source=to_source,
                                        )
                dist_to_data = dp.distanceToDataset(data_points=self._dataset._data_points,
                                                       selected_data_points=[list(self._dataset.trainingDataset().keys())],
                                                        )
                err_est_mesh[i,j] = reg_params[1] * dist_to_data + reg_params[0]

        ax2.set_title('Estimated Deviation [Rad]')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        cmap = matplotlib.cm.get_cmap(cmap_str)
        c = ax2.pcolormesh(az.cpu().detach().numpy(), elev.cpu().detach().numpy(), err_est_mesh.cpu().detach().numpy(), cmap=cmap, norm=norm)
        fig.colorbar(c, ax=ax2)

        if len(train_ax1) > 0:
            ax2.scatter(train_azim, train_elev, c=cmap(norm(train_err)), label='train', s=50, marker='s', edgecolors='black')
        if len(test_ax1) > 0:
            ax2.scatter(test_azim, test_elev, c=cmap(norm(test_err)), label='test', s=50, marker='D', edgecolors='black')
        if len(eval_ax1) > 0:
            ax2.scatter(eval_azim, eval_elev, c=cmap(norm(eval_err)), label='eval', marker='*', s=200, edgecolors='black')

        ax.set_ylabel('Deviation [Rad]')
        ax.set_xlabel('Hausdorff Distance [Rad]')
        ax.set_xlim((0,vmax_hausdorff))
        ax.set_ylim((0,vmax*1.05))
        ax.set_yticks(list(torch.arange(torch.tensor(0), torch.tensor(vmax), torch.tensor(0.001)).tolist()))
        ax.legend()

        ax2.set_ylabel('Source Elevation [°]')
        ax2.set_xlabel('Source Azimuth [°]')
        ax2.set_ylim((min_source_elev,max_source_elev))
        ax2.set_xlim((min_source_azim,max_source_azim))

        if show_plot:
            plt.show()

        if save_path:
            plt.savefig(save_path, format='png')
            plt.close()

# def plotDataDistributionOverAxesExample():
#     dataset = HeliOSHeliostatDataset()
#     file_path = '/Users/Synhelion/Downloads/calibdata.csv'
#     heliostat_list = ['AM.35']
#     dataset.loadDataFromFile(file_path=file_path, heliostat_list=heliostat_list)

#     mai = datetime.datetime(year=2022, month=2, day=1)
#     aug = datetime.datetime(year=2022, month=8, day=1)
#     sep = datetime.datetime(year=2022, month=9, day=1)
#     oct = datetime.datetime(year=2022, month=10, day=1)
#     nov = datetime.datetime(year=2022, month=11, day=1)

#     dataset.setTrainingData(created_at_range=[mai,sep])
#     dataset.setTestingData(created_at_range=[sep,oct])
#     dataset.setEvaluationData(created_at_range=[oct,nov])
#     dataset._updateHaussdorfDistances()

#     dataset_analyzer = HeliostatDatasetAnalyzer(dataset=dataset)

#     dataset_analyzer.plotDataDistributionOverAxes(plot_hausdorff=True)

# def plotDataDistributionOverDatesExample():
#     dataset = HeliOSHeliostatDataset()
#     file_path = '/Users/Synhelion/Downloads/calibdata.csv'
#     heliostat_list = ['AM.35']
#     dataset.loadDataFromFile(file_path=file_path, heliostat_list=heliostat_list)

#     mai = datetime.datetime(year=2022, month=2, day=1)
#     aug = datetime.datetime(year=2022, month=8, day=1)
#     sep = datetime.datetime(year=2022, month=9, day=1)
#     oct = datetime.datetime(year=2022, month=10, day=1)
#     nov = datetime.datetime(year=2022, month=11, day=1)

#     dataset.setTrainingData(created_at_range=[mai,sep])
#     dataset.setTestingData(created_at_range=[sep,oct])
#     dataset.setEvaluationData(created_at_range=[oct,nov])
#     dataset._updateHaussdorfDistances()

#     dataset.plotDataDistributionOverDates()

# if __name__ == '__main__':
#     plotDataDistributionOverAxesExample()
    # plotDataDistributionOverDatesExample()