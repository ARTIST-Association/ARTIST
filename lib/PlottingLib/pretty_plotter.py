import typing

import matplotlib as mpl
import matplotlib.pyplot as plt

class PlotData:
    
    def __init__(self,
                 x : typing.Any,
                 y : typing.Any,
                 s : typing.Optional[typing.Any] = None,
                 c : typing.Optional[str] = None,
                 m : typing.Optional[str] = None,
                 alpha : float = 1.0,
                 label : typing.Optional[str] = None,
                 adjust_size_by_value : typing.Optional[str] = None,
                 plot_type : str = 'line',
                 ) :
        self._x = x
        self._y = y
        self._s = s
        self._c = c
        self._m = m
        self._alpha = alpha
        self._label = label
        self._adjust_size_by_value = adjust_size_by_value
        self._plot_type = plot_type

    def label(self) -> typing.Optional[str] :
        return self._label

class ScientificPlotter:

    def __init__(self,
                 fig_width : float = 35.4,
                 fig_height : float = 10,
                 column_plot : bool = False,
                 title : typing.Optional[str] = None,
                 x_label : typing.List[str] = [],
                 y_label : typing.List[str] = [],
                 num_subplots = 1
                ):
        self._plot_data = []

        fig_width = fig_width / 2 if column_plot else fig_width

        self._fig, self._axs = plt.subplots(2)
        self._fig.set_figwidth = fig_width
        self._fig.set_figheight = fig_height

        for ax, i in enumerate(self._axs):
            j


    def addData(self, data : PlotData, subplot : int = ):
        self._plot_data.append(data)

    def addLegend(self, ax) -> bool :

        if any([d.label() for d in self._plot_data]):
            ax.legend(frameon=False)

    def create_figure(self, 
                      output_path : str, 
                      fig_height : typing.Optional[float] = None,
                      column_plot : bool = False,
                      title : typing.Optional[str] = None,
                      x_label : typing.Optional[str] = None,
                      y_label : typing.Optional[str] = None,
                      ):
        fig_height = fig_height if fig_height else self._fig_height
        fig_width = self._fig_width / 2 if column_plot else self._fig_width
        fig = plt.figure(figsize=(fig_width, fig_height))