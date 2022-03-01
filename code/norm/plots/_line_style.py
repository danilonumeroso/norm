from enum import Enum


class LineStyle(str, Enum):
    Simple = '-',
    Dashed = '--',
    Dotted = ':',
    Dashdot = '-.'
