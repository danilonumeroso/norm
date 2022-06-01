from __future__ import annotations

from ._line_style import LineStyle
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from numpy import array
from numpy.typing import NDArray

_ALPHA = 0.25

FloatArray = Union[List[float], NDArray[float]]

@dataclass(init=True)
class Line:
    x_data: List[float]
    y_data: List[float]
    std_error: Optional[List[float], NDArray[float]]
    color: str
    style: LineStyle
    label: str


@dataclass(init=True)
class Figure:
    name: str
    lines: List[Line]
    title: str
    theme: str
    x_label: str
    y_label: str
    x_ticks: List[float]
    y_ticks: List[float]
    x_lim: Tuple[int, int]
    y_lim: Tuple[int, int]
    x_scale: str
    y_scale: str
    has_legend: bool
    fontsize: Optional[str]
    dpi: int

    def add_line(self,
                 x_data: FloatArray,
                 y_data: FloatArray,
                 label: Optional[str] = "",
                 std_error: Optional[FloatArray] = None,
                 color: Optional[str] = None,
                 style: Optional[LineStyle] = LineStyle.Simple) -> Figure:

        assert len(x_data) == len(y_data), "Size mismatch"

        if std_error is not None:
            assert len(x_data) == len(std_error)

        self.lines.append(
            Line(array(x_data),
                 array(y_data),
                 array(std_error) if std_error is not None else None,
                 color,
                 style,
                 label)
        )

        return self  # allow chainable operations

    def finalize(self, show=True, save=True):
        from matplotlib import pyplot as plt
        from matplotlib.colors import to_rgba

        with plt.style.context(self.theme):
            # general config
            plt.title(self.title, fontsize=self.fontsize)
            plt.xticks(ticks=self.x_ticks)
            plt.yticks(ticks=self.y_ticks)
            plt.xlim(self.x_lim)
            plt.ylim(self.y_lim)
            plt.xlabel(self.x_label)
            plt.ylabel(self.y_label)

            # plot lines
            for line in self.lines:
                plt.plot(line.x_data, line.y_data,
                         line.color, linestyle=line.style.value,
                         label=line.label)
                if line.std_error is not None:
                    plt.fill_between(line.x_data,
                                     line.y_data - line.std_error,
                                     line.y_data + line.std_error,
                                     color=(*to_rgba(line.color)[:-1], _ALPHA))

            if self.has_legend:
                plt.legend(fontsize=self.fontsize)

            if self.y_scale == "log":
                plt.yscale(self.y_scale, base=2)

            if self.x_scale == "log":
                plt.xscale(self.x_scale, base=2)

            if save:
                plt.savefig(f'{self.name}.png', dpi=self.dpi)
            if show:
                plt.show()

            plt.close()


def new_figure(name: str,
               title: Optional[str] = "",
               theme: Optional[str] = "seaborn",
               x_label: Optional[str] = "",
               y_label: Optional[str] = "",
               x_ticks: Optional[List[float]] = None,
               y_ticks: Optional[List[float]] = None,
               x_lim: Optional[Tuple[int, int]] = None,
               y_lim: Optional[Tuple[int, int]] = None,
               x_scale: Optional[str] = "linear",
               y_scale: Optional[str] = "linear",
               has_legend: Optional[bool] = False,
               fontsize: Optional[str] = "x-large",
               dpi: Optional[int] = 300) -> Figure:

    return Figure(name=name,
                  title=title,
                  theme=theme,
                  x_label=x_label,
                  y_label=y_label,
                  x_ticks=x_ticks,
                  y_ticks=y_ticks,
                  x_lim=x_lim if x_lim else (None, None),
                  y_lim=y_lim if y_lim else (None, None),
                  x_scale=x_scale,
                  y_scale=y_scale,
                  has_legend=has_legend,
                  fontsize=fontsize,
                  dpi=dpi,
                  lines=[])
