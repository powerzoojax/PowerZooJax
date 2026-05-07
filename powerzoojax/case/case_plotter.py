"""
CasePlotter: Network Visualization (CPU-only)

Provides visualization utilities for power system networks.
Complete migration from PowerZoo's CasePlotter.

⚠️ WARNING: All methods run on CPU and transfer data from GPU.
Do NOT call plotting functions inside training loops.

Usage:
    >>> from powerzoojax.case import create_case5
    >>> from powerzoojax.case.case_plotter import CasePlotter
    >>> 
    >>> case = create_case5()
    >>> plotter = CasePlotter(case)
    >>> fig, ax = plotter.plot_topology()
    >>> plt.show()
"""
from __future__ import annotations

import warnings
from typing import Optional, Dict, Tuple, Any, List
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None  # type: ignore
    NETWORKX_AVAILABLE = False

from powerzoojax.case.case_data import CaseData


def _warn_cpu():
    """Print CPU warning."""
    print("⚠️  [CPU] Plotting requires CPU - transferring data from GPU")


def _to_numpy(arr) -> np.ndarray:
    """Convert JAX array to numpy."""
    if arr is None:
        return None
    if hasattr(arr, 'to_py'):
        return np.array(arr.to_py())
    return np.array(arr)


class CasePlotter:
    """Plotter for power system case visualization.
    
    Provides methods to visualize network topology and power flow results.
    All operations run on CPU.
    
    ⚠️ WARNING: Do not use in training loops.
    
    Attributes:
        case: The CaseData object
        G: NetworkX graph representation
        pos: Node positions for plotting
    """
    
    def __init__(self, case: CaseData, warn: bool = True):
        """Initialize plotter with a CaseData.
        
        Args:
            case: CaseData to visualize
            warn: Whether to show CPU warnings
        """
        if not MATPLOTLIB_AVAILABLE or not NETWORKX_AVAILABLE:
            raise ImportError("matplotlib and networkx are required for plotting")
        
        if warn:
            _warn_cpu()
        
        self.case = case
        self._warn = warn
        self.G = None
        self.pos = None
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build NetworkX graph from case data."""
        self.G = nx.Graph()
        
        # Convert to numpy
        node_ids = _to_numpy(self.case.node_ids)
        node_x = _to_numpy(self.case.node_x)
        node_y = _to_numpy(self.case.node_y)
        line_from = _to_numpy(self.case.line_from)
        line_to = _to_numpy(self.case.line_to)
        
        # Add nodes
        for i, nid in enumerate(node_ids):
            self.G.add_node(
                int(nid),
                x=float(node_x[i]) if node_x is not None else 0,
                y=float(node_y[i]) if node_y is not None else 0
            )
        
        # Add edges
        for i in range(len(line_from)):
            self.G.add_edge(int(line_from[i]), int(line_to[i]))
    
    def _compute_layout(self, layout: str = 'auto') -> Dict[int, Tuple[float, float]]:
        """Compute node positions for plotting.
        
        Args:
            layout: Layout algorithm ('auto', 'spring', 'kamada_kawai', 'case', 'radial', 'feeder')
        
        Returns:
            Dictionary mapping node IDs to (x, y) positions
        """
        if layout == 'case':
            # Use case-defined coordinates
            self.pos = {
                int(nid): (self.G.nodes[int(nid)]['x'], self.G.nodes[int(nid)]['y'])
                for nid in self.G.nodes()
            }
        elif layout == 'auto':
            # Check if case has meaningful coordinates
            coords = list(self.pos.values()) if self.pos else [(0, 0)]
            has_coords = not all(c == (0, 0) for c in coords)
            
            if has_coords:
                layout = 'case'
            elif nx.is_tree(self.G):
                layout = 'feeder'
            else:
                layout = 'kamada_kawai'
            return self._compute_layout(layout)
        elif layout == 'spring':
            self.pos = nx.spring_layout(self.G, seed=42, k=2.0, iterations=100)
        elif layout == 'kamada_kawai':
            self.pos = nx.kamada_kawai_layout(self.G)
        elif layout == 'spectral':
            self.pos = nx.spectral_layout(self.G)
        elif layout == 'radial' or layout == 'feeder':
            # Radial tree layout for distribution systems
            root = min(self.G.nodes())
            try:
                bfs_tree = nx.bfs_tree(self.G, root)
                self.pos = self._feeder_layout(bfs_tree, root)
            except:
                self.pos = nx.kamada_kawai_layout(self.G)
        else:
            self.pos = nx.spring_layout(self.G, seed=42)
        
        return self.pos
    
    def _feeder_layout(self, tree: nx.DiGraph, root: int) -> Dict[int, Tuple[float, float]]:
        """Compute feeder layout optimized for radial networks.
        
        Places main feeder horizontally, with branches going up/down.
        """
        pos = {}
        
        # Find the main trunk (longest path from root)
        def find_longest_path(node, path=[]):
            path = path + [node]
            children = list(tree.successors(node))
            if not children:
                return path
            
            longest = path
            for child in children:
                child_path = find_longest_path(child, path)
                if len(child_path) > len(longest):
                    longest = child_path
            return longest
        
        main_trunk = find_longest_path(root)
        main_trunk_set = set(main_trunk)
        
        # Position main trunk horizontally
        for i, node in enumerate(main_trunk):
            pos[node] = (i * 1.2, 0)
        
        # Position branches
        branch_direction = 1
        
        for trunk_node in main_trunk:
            children = list(tree.successors(trunk_node))
            branch_children = [c for c in children if c not in main_trunk_set]
            
            for branch_start in branch_children:
                direction = branch_direction
                branch_direction *= -1
                
                trunk_x = pos[trunk_node][0]
                
                # BFS within branch
                branch_levels = {branch_start: 0}
                queue = [branch_start]
                while queue:
                    node = queue.pop(0)
                    for child in tree.successors(node):
                        if child not in branch_levels and child not in main_trunk_set:
                            branch_levels[child] = branch_levels[node] + 1
                            queue.append(child)
                
                # Position branch nodes
                for node, level in branch_levels.items():
                    y = direction * (1.5 + level * 1.0)
                    x = trunk_x + (level + 1) * 1.2
                    pos[node] = (x, y)
        
        return pos
    
    def plot_topology(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (14, 10),
        layout: str = 'auto',
        node_size: int = 500,
        node_color: str = '#4A90D9',
        edge_color: str = '#666666',
        edge_width: float = 1.5,
        show_node_labels: bool = True,
        title: str = None,
        highlight_generators: bool = True,
        highlight_loads: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot network topology.
        
        Args:
            ax: Matplotlib axes (creates new figure if None)
            figsize: Figure size (width, height)
            layout: Layout algorithm ('auto', 'case', 'spring', 'kamada_kawai', 'feeder')
            node_size: Size of nodes
            node_color: Default color for buses
            edge_color: Color of edges
            edge_width: Width of edges
            show_node_labels: Whether to show node ID labels
            title: Plot title
            highlight_generators: Show generator buses differently
            highlight_loads: Scale node size by load
        
        Returns:
            (figure, axes) tuple
        """
        if self._warn:
            _warn_cpu()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize,dpi=300)
        else:
            fig = ax.get_figure()
        
        # Compute layout
        self._compute_layout(layout)
        
        # Node colors
        unit_bus_ids = set(_to_numpy(self.case.unit_bus_ids).astype(int))
        node_colors = []
        for node in self.G.nodes():
            if highlight_generators and node in unit_bus_ids:
                if node == int(_to_numpy(self.case.node_ids)[self.case.slack_bus_idx]):
                    node_colors.append('#E74C3C')  # Red for slack
                else:
                    node_colors.append('#27AE60')  # Green for generators
            else:
                node_colors.append(node_color)
        
        # Node sizes based on load
        if highlight_loads:
            load_bus_ids = _to_numpy(self.case.load_bus_ids).astype(int)
            load_d_max = _to_numpy(self.case.load_d_max)
            node_loads = {int(bid): load_d_max[i] for i, bid in enumerate(load_bus_ids)}
            max_load = max(node_loads.values()) if node_loads else 1
            
            node_sizes = []
            for node in self.G.nodes():
                load = node_loads.get(node, 0)
                size = node_size * (0.5 + 1.5 * load / max_load)
                node_sizes.append(size)
        else:
            node_sizes = [node_size] * len(self.G.nodes())
        
        # Draw edges
        nx.draw_networkx_edges(
            self.G, self.pos, ax=ax,
            edge_color=edge_color,
            width=edge_width,
            alpha=0.7
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.G, self.pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='white',
            linewidths=2
        )
        
        # Draw labels
        if show_node_labels:
            nx.draw_networkx_labels(
                self.G, self.pos, ax=ax,
                font_size=9,
                font_color='white',
                font_weight='bold'
            )
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='#E74C3C', label='Slack Bus'),
            mpatches.Patch(color='#27AE60', label='Generator'),
            mpatches.Patch(color=node_color, label='Load Bus'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        # else:
        #     ax.set_title("Network Topology", fontsize=14, fontweight='bold')
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig, ax
    
    def plot_line_flows(
        self,
        line_flows: np.ndarray,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (14, 10),
        layout: str = 'auto',
        colormap: str = 'coolwarm',
        show_values: bool = True,
        title: str = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot line power flows on network.
        
        Args:
            line_flows: Power flow on each line (n_lines,) [MW]
            ax: Matplotlib axes
            figsize: Figure size
            layout: Layout algorithm
            colormap: Colormap for flow values
            show_values: Show flow values on edges
            title: Plot title
        
        Returns:
            (figure, axes) tuple
        """
        if self._warn:
            _warn_cpu()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        # Ensure numpy
        line_flows = _to_numpy(line_flows)
        
        # Compute layout
        self._compute_layout(layout)
        
        # Edge colors based on flow
        cmap = plt.get_cmap(colormap)
        max_flow = np.max(np.abs(line_flows)) if len(line_flows) > 0 else 1
        norm = mcolors.Normalize(vmin=-max_flow, vmax=max_flow)
        
        line_from = _to_numpy(self.case.line_from).astype(int)
        line_to = _to_numpy(self.case.line_to).astype(int)
        
        edge_colors = []
        edge_widths = []
        for i, (f, t) in enumerate(zip(line_from, line_to)):
            flow = line_flows[i] if i < len(line_flows) else 0
            edge_colors.append(cmap(norm(flow)))
            edge_widths.append(1 + 3 * abs(flow) / max_flow)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.G, self.pos, ax=ax,
            node_color='#4A90D9',
            node_size=400,
            alpha=0.9
        )
        
        # Draw edges with flow colors
        edges = list(zip(line_from, line_to))
        nx.draw_networkx_edges(
            self.G, self.pos, ax=ax,
            edgelist=edges,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.8
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            self.G, self.pos, ax=ax,
            font_size=8,
            font_color='white'
        )
        
        # Add flow values
        if show_values:
            for i, (f, t) in enumerate(zip(line_from, line_to)):
                if i < len(line_flows):
                    x = (self.pos[f][0] + self.pos[t][0]) / 2
                    y = (self.pos[f][1] + self.pos[t][1]) / 2
                    ax.annotate(
                        f'{line_flows[i]:.1f}',
                        xy=(x, y),
                        fontsize=7,
                        ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
                    )
        
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('Power Flow (MW)', fontsize=10)
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title("Line Power Flows", fontsize=14, fontweight='bold')
        
        ax.set_aspect('equal')
        ax.axis('off')
        
        return fig, ax
    
    def plot_ptdf_heatmap(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (12, 8),
        colormap: str = 'RdBu_r',
        title: str = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot PTDF matrix as heatmap.
        
        Args:
            ax: Matplotlib axes
            figsize: Figure size
            colormap: Colormap
            title: Plot title
        
        Returns:
            (figure, axes) tuple
        """
        if self._warn:
            _warn_cpu()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        
        PTDF = _to_numpy(self.case.PTDF)
        
        im = ax.imshow(PTDF, cmap=colormap, aspect='auto')
        
        # Labels
        ax.set_xlabel('Node Index', fontsize=11)
        ax.set_ylabel('Line Index', fontsize=11)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('PTDF Value', fontsize=10)
        
        # Title
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title("PTDF Matrix", fontsize=14, fontweight='bold')
        
        return fig, ax


def plot_case(case: CaseData, **kwargs: Any) -> Tuple[plt.Figure, plt.Axes]:
    """Convenience function to plot case topology.
    
    Args:
        case: CaseData to plot
        **kwargs: Arguments passed to CasePlotter.plot_topology
    
    Returns:
        (figure, axes) tuple
    """
    plotter = CasePlotter(case)
    return plotter.plot_topology(**kwargs)
