#!/usr/bin/env python3
"""
Centralized Color Configuration for All Plotting Scripts
Style: Anthropic Research Palette

This module defines consistent colors for BFN Policy and Diffusion Policy
across all publication figures.

Usage:
    from colors import COLORS, setup_matplotlib_style, ANTHROPIC_CMAP
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# =============================================================================
# ANTHROPIC COLOR PALETTE
# =============================================================================

# Primary Colors
ANTHROPIC_CORAL = '#D97757'      # RGB(217, 119, 87)  - Warm, distinctive
ANTHROPIC_TAN = '#C4A77D'        # RGB(196, 167, 125) - Earthy accent
ANTHROPIC_SLATE = '#4A4A4A'      # RGB(74, 74, 74)    - Professional dark
ANTHROPIC_SAND = '#F5F0E8'       # RGB(245, 240, 232) - Warm background
ANTHROPIC_TEAL = '#2D8B8B'       # RGB(45, 139, 139)  - Cool accent

# Light versions (for fills, backgrounds)
ANTHROPIC_CORAL_LT = '#F3C8B4'   # RGB(243, 200, 180)
ANTHROPIC_TEAL_LT = '#B4D7D7'    # RGB(180, 215, 215)
ANTHROPIC_TAN_LT = '#EBDFC8'     # RGB(235, 223, 200)
ANTHROPIC_CREAM = '#FCFAF5'      # RGB(252, 250, 245)

# Dark versions (for emphasis)
ANTHROPIC_CORAL_DK = '#B85A3D'   # RGB(184, 90, 61)
ANTHROPIC_TEAL_DK = '#1E6B6B'    # RGB(30, 107, 107)

# =============================================================================
# SEMANTIC COLOR MAPPING (Use these in all scripts!)
# =============================================================================

COLORS = {
    # Method colors (ALWAYS USE THESE FOR BFN/DIFFUSION)
    'bfn': ANTHROPIC_TEAL,              # BFN Policy - Teal (Ours)
    'diffusion': ANTHROPIC_CORAL,       # Diffusion Policy - Coral (Baseline)
    
    # Light versions for fills/confidence bands
    'bfn_light': ANTHROPIC_TEAL_LT,
    'diffusion_light': ANTHROPIC_CORAL_LT,
    
    # Dark versions for emphasis/borders
    'bfn_dark': ANTHROPIC_TEAL_DK,
    'diffusion_dark': ANTHROPIC_CORAL_DK,
    
    # Semantic colors
    'ground_truth': ANTHROPIC_SLATE,    # Ground truth data
    'neutral': ANTHROPIC_TAN,           # Neutral elements
    'highlight': ANTHROPIC_TAN,         # Highlights, motion arrows
    
    # UI colors
    'text': ANTHROPIC_SLATE,            # Text, labels
    'gray': ANTHROPIC_SLATE,            # Gray elements
    'light_gray': ANTHROPIC_SAND,       # Light backgrounds
    'background': ANTHROPIC_SAND,       # Panel backgrounds
    'cream': ANTHROPIC_CREAM,           # Very light background
    'black': '#333333',                 # Near black
    'white': '#FFFFFF',                 # White
    
    # Additional semantic
    'success': ANTHROPIC_TEAL,          # Success/positive
    'warning': ANTHROPIC_CORAL,         # Warning/attention
    'error': ANTHROPIC_CORAL_DK,        # Error states
    
    # Legacy/compatibility colors (map to Anthropic equivalents)
    'light_green': ANTHROPIC_TEAL_LT,   # Used for target zones
    'light_blue': ANTHROPIC_TEAL_LT,    # Used for various highlights
    'light_orange': ANTHROPIC_CORAL_LT, # Used for various highlights
    'orange': ANTHROPIC_CORAL,          # Alias for diffusion color
    'blue': ANTHROPIC_TEAL,             # Alias for bfn color
}

# =============================================================================
# CUSTOM COLORMAPS
# =============================================================================

# BFN colormap (light teal → teal → dark teal)
CMAP_BFN = LinearSegmentedColormap.from_list(
    'anthropic_bfn', 
    [ANTHROPIC_TEAL_LT, ANTHROPIC_TEAL, ANTHROPIC_TEAL_DK]
)

# Diffusion colormap (light coral → coral → dark coral)
CMAP_DIFFUSION = LinearSegmentedColormap.from_list(
    'anthropic_diffusion',
    [ANTHROPIC_CORAL_LT, ANTHROPIC_CORAL, ANTHROPIC_CORAL_DK]
)

# Time colormap (sand → tan → slate)
CMAP_TIME = LinearSegmentedColormap.from_list(
    'anthropic_time',
    [ANTHROPIC_CREAM, ANTHROPIC_TAN, ANTHROPIC_SLATE]
)

# Sequential colormap (cream → teal)
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    'anthropic_seq',
    [ANTHROPIC_CREAM, ANTHROPIC_TEAL_LT, ANTHROPIC_TEAL]
)

# Diverging colormap (coral → cream → teal)
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    'anthropic_div',
    [ANTHROPIC_CORAL, ANTHROPIC_CREAM, ANTHROPIC_TEAL]
)

# Alias for backward compatibility
ANTHROPIC_CMAP = {
    'bfn': CMAP_BFN,
    'diffusion': CMAP_DIFFUSION,
    'time': CMAP_TIME,
    'sequential': CMAP_SEQUENTIAL,
    'diverging': CMAP_DIVERGING,
}

# =============================================================================
# FIGURE SIZES (Publication standards)
# =============================================================================

SINGLE_COL = 3.5    # inches - single column width (NeurIPS/ICML)
DOUBLE_COL = 7.0    # inches - double column width
ASPECT = 0.75       # default height/width ratio

# =============================================================================
# MATPLOTLIB STYLE SETUP
# =============================================================================

def setup_matplotlib_style():
    """
    Configure matplotlib with Anthropic-style settings.
    Call this at the start of any plotting script.
    
    Compatible with matplotlib 2.x and 3.x.
    """
    import matplotlib
    
    # Use Agg backend to avoid DISPLAY issues on headless servers
    matplotlib.use('Agg')
    
    # Base settings (compatible with older matplotlib)
    style_dict = {
        # Font settings (serif for academic papers)
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino', 'Georgia'],
        'mathtext.fontset': 'stix',
        
        # Font sizes
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        
        # Axes settings
        'axes.linewidth': 0.8,
        'axes.edgecolor': COLORS['gray'],
        'axes.labelcolor': COLORS['text'],
        'axes.facecolor': 'white',
        
        # Grid settings
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        
        # Tick settings
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.color': COLORS['gray'],
        'ytick.color': COLORS['gray'],
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.fancybox': False,
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
    }
    
    # Add version-specific settings
    mpl_version = tuple(int(x) for x in matplotlib.__version__.split('.')[:2])
    
    if mpl_version >= (3, 0):
        # Matplotlib 3.x specific settings
        style_dict.update({
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.titlesize': 11,
            'savefig.pad_inches': 0.02,
            'legend.edgecolor': COLORS['light_gray'],
            'grid.color': COLORS['gray'],
        })
    else:
        # Matplotlib 2.x fallback
        style_dict.update({
            'axes.spines.top': True,  # Can't disable in 2.x via rcParams
            'axes.spines.right': True,
        })
    
    plt.rcParams.update(style_dict)


def get_method_style(method: str) -> dict:
    """
    Get consistent plotting style for a method.
    
    Args:
        method: 'bfn' or 'diffusion'
    
    Returns:
        dict with color, light_color, marker, linestyle
    """
    if method.lower() == 'bfn':
        return {
            'color': COLORS['bfn'],
            'light_color': COLORS['bfn_light'],
            'dark_color': COLORS['bfn_dark'],
            'marker': 'o',
            'linestyle': '-',
            'label': 'BFN Policy (Ours)',
        }
    elif method.lower() == 'diffusion':
        return {
            'color': COLORS['diffusion'],
            'light_color': COLORS['diffusion_light'],
            'dark_color': COLORS['diffusion_dark'],
            'marker': 's',
            'linestyle': '-',
            'label': 'Diffusion Policy',
        }
    else:
        return {
            'color': COLORS['neutral'],
            'light_color': COLORS['light_gray'],
            'dark_color': COLORS['gray'],
            'marker': '^',
            'linestyle': '--',
            'label': method,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def annotate_speedup(ax, x1, y1, x2, y2, speedup_text, color=None):
    """Add a speedup annotation arrow between two points."""
    if color is None:
        color = COLORS['gray']
    
    ax.annotate('', xy=(x1, y1), xytext=(x2, y2),
                arrowprops=dict(arrowstyle='<->', color=color, lw=1.5, ls='--'))
    ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.02, speedup_text,
            ha='center', va='bottom', fontsize=8, fontweight='bold', color=COLORS['bfn'])


def create_legend_handles():
    """Create standard legend handles for BFN and Diffusion."""
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    return [
        Line2D([0], [0], color=COLORS['bfn'], linewidth=2, marker='o', 
               markersize=5, label='BFN Policy (Ours)'),
        Line2D([0], [0], color=COLORS['diffusion'], linewidth=2, marker='s',
               markersize=5, label='Diffusion Policy'),
    ]


# =============================================================================
# PRINT COLOR INFO (for debugging)
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("ANTHROPIC COLOR PALETTE FOR BFN VS DIFFUSION")
    print("=" * 60)
    print()
    print("Primary Method Colors:")
    print(f"  BFN Policy:       {COLORS['bfn']} (Teal)")
    print(f"  Diffusion Policy: {COLORS['diffusion']} (Coral)")
    print()
    print("Light Fills:")
    print(f"  BFN Light:        {COLORS['bfn_light']}")
    print(f"  Diffusion Light:  {COLORS['diffusion_light']}")
    print()
    print("Neutral Colors:")
    print(f"  Ground Truth:     {COLORS['ground_truth']}")
    print(f"  Background:       {COLORS['background']}")
    print(f"  Text:             {COLORS['text']}")
    print()
    print("Usage in scripts:")
    print("  from colors import COLORS, setup_matplotlib_style")
    print("  setup_matplotlib_style()")
    print("  plt.plot(x, y, color=COLORS['bfn'], label='BFN Policy')")

