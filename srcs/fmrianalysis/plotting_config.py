"""
Configuration for brain surface plotting
"""

# Plotting parameters
PLOT_PARAMS = {
    'threshold': 2.3,
    'cmap': 'RdBu_r',
    'bg_on_data': True,
    'colorbar': False,
    'vmax': 7.3,
    'symmetric_cmap': True
}

# Figure parameters
FIGURE_PARAMS = {
    'figsize': (20, 5),
    'dpi': 300,
    'facecolor': 'white',
    'edgecolor': 'none'
}

# Colorbar parameters
COLORBAR_PARAMS = {
    'position': [0.90, 0.25, 0.02, 0.5],  # [left, bottom, width, height]
    'label': 'z-score',
    'label_size': 12,
    'tick_size': 10,
    'gray_color': [0.5, 0.5, 0.5, 1.0]  # RGBA for sub-threshold regions
}

# Layout parameters
LAYOUT_PARAMS = {
    'left': 0.02,
    'right': 0.88,
    'top': 0.9,
    'bottom': 0.1,
    'wspace': 0.05
}