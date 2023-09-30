import matplotlib.pyplot as plt
import matplotlib


def update_plot_style():
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.size': 16.0,
        'lines.linewidth': 2,
        'lines.antialiased': True,
        'axes.facecolor': 'fdfdfd',
        'axes.edgecolor': '777777',
        'axes.linewidth': 1,
        'axes.titlesize': 'medium',
        'axes.labelsize': 'medium',
        'axes.axisbelow': True,
        'xtick.major.size': 0,  # major tick size in points
        'xtick.minor.size': 0,  # minor tick size in points
        'xtick.major.pad': 6,  # distance to major tick label in points
        'xtick.minor.pad': 6,  # distance to the minor tick label in points
        'xtick.color': '333333',  # color of the tick labels
        'xtick.labelsize': 'medium',  # fontsize of the tick labels
        'xtick.direction': 'in',  # direction: in or out
        'ytick.major.size': 0,  # major tick size in points
        'ytick.minor.size': 0,  # minor tick size in points
        'ytick.major.pad': 6,  # distance to major tick label in points
        'ytick.minor.pad': 6,  # distance to the minor tick label in points
        'ytick.color': '333333',  # color of the tick labels
        'ytick.labelsize': 'medium',  # fontsize of the tick labels
        'ytick.direction': 'in',  # direction: in or out
        'axes.grid': False,
        'grid.alpha': 0.3,
        'grid.linewidth': 1,
        'legend.fancybox': True,
        'legend.fontsize': 'Small',
        'figure.figsize': (5, 4),
        'figure.facecolor': '1.0',
        'figure.edgecolor': '0.5',
        'hatch.linewidth': 0.1,
        'text.usetex': False
    })


def get_color_map():
    color_map = {'green': '#009E60', 'orange': '#C04000',
                 'blue': 'C0', 'black': '#3A3B3C',
                 'purple': '#843B62', 'red': '#C41E3A'}
    return color_map