"""
CaseInfo: DataFrame Printing and CPU Utilities

Provides PowerZoo-compatible DataFrame printing format for CaseData.
All methods in this module run on CPU and transfer data from GPU.

⚠️ WARNING: Methods in this module transfer data from GPU to CPU.
Do NOT call these methods inside training loops - use only for debugging.

Usage:
    >>> from powerzoojax.case import create_case5
    >>> from powerzoojax.case.case_info import CaseInfo
    >>> 
    >>> case_data = create_case5()
    >>> info = CaseInfo(case_data)
    >>> print(info)  # PowerZoo-style DataFrame printing
"""

import warnings
from typing import Optional, List, Any
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from powerzoojax.case.case_data import CaseData


def _warn_cpu_transfer(operation: str = "operation"):
    """Print warning about CPU data transfer."""
    print(f"⚠️  [CPU] {operation} - transferring data from GPU to CPU")


def _jax_to_numpy(arr) -> np.ndarray:
    """Convert JAX array to numpy, handling None gracefully."""
    if arr is None:
        return None
    # Use numpy() if available (JAX arrays), otherwise try np.array
    if hasattr(arr, 'to_py'):
        return np.array(arr.to_py())
    return np.array(arr)


class CaseDataFrame:
    """DataFrame-like wrapper for case data tables.
    
    Mimics PowerZoo's DataFrame display format.
    """
    
    def __init__(self, data: np.ndarray, columns: List[str], label: str = ""):
        """Initialize with numpy array and column names.
        
        Args:
            data: 2D numpy array
            columns: Column names
            label: Table label for display
        """
        self.data = data
        self.columns = columns
        self.label = label
        self._df = None
    
    @property
    def df(self):
        """Lazy create pandas DataFrame."""
        if self._df is None and PANDAS_AVAILABLE:
            self._df = pd.DataFrame(self.data, columns=self.columns)
            if 'id' in self.columns:
                self._df.index = self._df['id'].astype(int)
        return self._df
    
    def __repr__(self) -> str:
        """PowerZoo-compatible string representation."""
        header = f"{'=' * 20} {self.label} {'=' * 20}"
        
        if PANDAS_AVAILABLE and self.df is not None:
            return f"{header}\n{self.df.to_string(index=False)}"
        else:
            # Fallback without pandas
            lines = [header]
            # Header row
            lines.append("  ".join(f"{col:>10}" for col in self.columns))
            # Data rows
            for row in self.data:
                lines.append("  ".join(f"{val:>10.4f}" if isinstance(val, float) else f"{val:>10}" for val in row))
            return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.__repr__()


class CaseInfo:
    """CPU-side utilities for CaseData inspection and printing.
    
    Provides PowerZoo-compatible printing format and data inspection.
    
    ⚠️ WARNING: All methods transfer data from GPU to CPU.
    Use only for debugging, NOT in training loops.
    
    Attributes:
        case_data: The underlying CaseData
        name: Case name
        
    Example:
        >>> case = create_case5()
        >>> info = CaseInfo(case)
        >>> print(info)  # Print all tables
        >>> print(info.nodes)  # Print nodes table only
    """
    
    def __init__(
        self,
        case_data: CaseData,
        warn: bool = True,
        name: Optional[str] = None,
    ):
        """Initialize CaseInfo wrapper.
        
        Args:
            case_data: CaseData to wrap
            warn: Whether to show CPU transfer warnings (default True)
            name: Display name (``CaseData`` has no ``name`` field for JIT safety)
        """
        self._case = case_data
        self._warn = warn
        self.name = name if name is not None else "CaseData"
        
        # Lazy-loaded tables (converted to numpy on first access)
        self._nodes_df = None
        self._units_df = None
        self._lines_df = None
        self._loads_df = None
    
    def _maybe_warn(self, operation: str):
        """Print warning if enabled."""
        if self._warn:
            _warn_cpu_transfer(operation)
    
    @property
    def nodes(self) -> CaseDataFrame:
        """Get nodes table as CaseDataFrame."""
        if self._nodes_df is None:
            self._maybe_warn("Accessing nodes table")
            
            # Build nodes data array
            n = self._case.n_nodes
            data = np.column_stack([
                _jax_to_numpy(self._case.node_ids),
                _jax_to_numpy(self._case.node_x),
                _jax_to_numpy(self._case.node_y),
            ])
            
            self._nodes_df = CaseDataFrame(
                data=data,
                columns=['id', 'x', 'y'],
                label='nodes'
            )
        return self._nodes_df
    
    @property
    def units(self) -> CaseDataFrame:
        """Get units/generators table as CaseDataFrame."""
        if self._units_df is None:
            self._maybe_warn("Accessing units table")
            
            data = np.column_stack([
                _jax_to_numpy(self._case.unit_ids),
                _jax_to_numpy(self._case.unit_bus_ids),
                _jax_to_numpy(self._case.unit_cost_a),
                _jax_to_numpy(self._case.unit_cost_b),
                _jax_to_numpy(self._case.unit_cost_c),
                _jax_to_numpy(self._case.unit_p_max),
                _jax_to_numpy(self._case.unit_p_min),
            ])
            
            self._units_df = CaseDataFrame(
                data=data,
                columns=['id', 'bus_id', 'mc_a', 'mc_b', 'mc_c', 'p_max', 'p_min'],
                label='units'
            )
        return self._units_df
    
    @property
    def lines(self) -> CaseDataFrame:
        """Get lines/branches table as CaseDataFrame."""
        if self._lines_df is None:
            self._maybe_warn("Accessing lines table")
            
            data = np.column_stack([
                _jax_to_numpy(self._case.line_ids),
                _jax_to_numpy(self._case.line_from),
                _jax_to_numpy(self._case.line_to),
                _jax_to_numpy(self._case.line_x),
                _jax_to_numpy(self._case.line_floor),
                _jax_to_numpy(self._case.line_cap),
            ])
            
            self._lines_df = CaseDataFrame(
                data=data,
                columns=['id', 'from', 'to', 'x', 'floor', 'cap'],
                label='lines'
            )
        return self._lines_df
    
    @property
    def loads(self) -> CaseDataFrame:
        """Get loads table as CaseDataFrame."""
        if self._loads_df is None:
            self._maybe_warn("Accessing loads table")
            
            data = np.column_stack([
                _jax_to_numpy(self._case.load_ids),
                _jax_to_numpy(self._case.load_bus_ids),
                _jax_to_numpy(self._case.load_d_max),
                _jax_to_numpy(self._case.load_d_min),
            ])
            
            self._loads_df = CaseDataFrame(
                data=data,
                columns=['id', 'bus_id', 'd_max', 'd_min'],
                label='loads'
            )
        return self._loads_df
    
    def __repr__(self) -> str:
        """PowerZoo-compatible full case representation."""
        self._maybe_warn("Printing full case")
        
        lines = [
            f"Case: {self.name}",
            f"Nodes: {self._case.n_nodes}, Lines: {self._case.n_lines}, "
            f"Units: {self._case.n_units}, Loads: {self._case.n_loads}",
            "",
            str(self.nodes),
            "",
            str(self.units),
            "",
            str(self.lines),
            "",
            str(self.loads),
        ]
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def summary(self) -> str:
        """Get a brief summary without full tables."""
        self._maybe_warn("Getting summary")
        
        total_gen = float(_jax_to_numpy(self._case.unit_p_max).sum())
        total_load = float(_jax_to_numpy(self._case.load_d_max).sum())
        
        return (
            f"Case: {self.name}\n"
            f"  Nodes: {self._case.n_nodes}\n"
            f"  Lines: {self._case.n_lines}\n"
            f"  Units: {self._case.n_units} (total capacity: {total_gen:.2f} MW)\n"
            f"  Loads: {self._case.n_loads} (total demand: {total_load:.2f} MW)\n"
            f"  Slack bus index: {self._case.slack_bus_idx}\n"
            f"  PTDF shape: {self._case.PTDF.shape if self._case.PTDF is not None else 'Not computed'}"
        )
    
    def get_ptdf_numpy(self) -> np.ndarray:
        """Get PTDF matrix as numpy array.
        
        ⚠️ Transfers from GPU to CPU.
        """
        self._maybe_warn("Getting PTDF matrix")
        return _jax_to_numpy(self._case.PTDF)
    
    def get_adjacency_numpy(self) -> np.ndarray:
        """Compute and return adjacency matrix as numpy array.
        
        ⚠️ Runs on CPU.
        """
        self._maybe_warn("Computing adjacency matrix")
        
        n = self._case.n_nodes
        line_from_idx = _jax_to_numpy(self._case.line_from_idx)
        line_to_idx = _jax_to_numpy(self._case.line_to_idx)
        
        A = np.zeros((n, n))
        A[line_from_idx, line_to_idx] = 1.0
        A[line_to_idx, line_from_idx] = 1.0
        
        return A


def print_case(case_data: CaseData, warn: bool = True):
    """Convenience function to print a CaseData.
    
    Args:
        case_data: CaseData to print
        warn: Whether to show CPU transfer warning
    """
    info = CaseInfo(case_data, warn=warn)
    print(info)


def print_summary(case_data: CaseData, warn: bool = True):
    """Convenience function to print case summary.
    
    Args:
        case_data: CaseData to summarize
        warn: Whether to show CPU transfer warning
    """
    info = CaseInfo(case_data, warn=warn)
    print(info.summary())
