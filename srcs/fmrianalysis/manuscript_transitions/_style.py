"""Shared manuscript-figure style helpers for manuscript_transitions.

Keeps the three figures visually consistent and honours project conventions:
  - tick label zero renders as "0", never "0.0"/"0.00"  (see memory)
"""
import matplotlib.ticker as mticker


def _zero_fmt(x, _pos):
    """Format tick: 0 -> '0'; otherwise trim trailing zeros sensibly."""
    if abs(x) < 1e-9:
        return '0'
    s = f'{x:.2f}'.rstrip('0').rstrip('.')
    return s


ZERO_FORMATTER = mticker.FuncFormatter(_zero_fmt)


def apply_zero_format(*axes, which='y'):
    """Apply the zero-formatter to the given axes' numeric axis/axes."""
    for ax in axes:
        if ax is None:
            continue
        if which in ('y', 'both'):
            ax.yaxis.set_major_formatter(ZERO_FORMATTER)
        if which in ('x', 'both'):
            ax.xaxis.set_major_formatter(ZERO_FORMATTER)
