#!/usr/bin/env python3
"""Convert PowerZoo case data files to PowerZooJax native JAX format.

Reads each PowerZoo Case class, extracts the DataFrame data, and writes
a create_case*() -> CaseData function with embedded jnp arrays.

Usage:
    python scripts/convert_cases.py
"""

import sys
import os
import textwrap
import numpy as np

sys.path.insert(0, os.path.expanduser("~/codes/PowerZoo"))

from powerzoo.case.CaseBase import ClearCase

FUEL_MAP = {'nuclear': 1, 'coal': 2, 'gas': 3, 'oil': 4, 'hydro': 5, 'wind': 6, 'solar': 7}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'powerzoojax', 'case', 'cases')


def fmt_array(arr, name, indent=4, per_line=10):
    """Format a numpy array as a jnp.array(...) string."""
    prefix = " " * indent
    vals = arr.tolist()
    if len(vals) <= per_line:
        val_str = ", ".join(_fmt_val(v) for v in vals)
        return f"{prefix}{name} = jnp.array([{val_str}])"
    lines = []
    lines.append(f"{prefix}{name} = jnp.array([")
    for i in range(0, len(vals), per_line):
        chunk = vals[i:i+per_line]
        chunk_str = ", ".join(_fmt_val(v) for v in chunk)
        lines.append(f"{prefix}    {chunk_str},")
    lines.append(f"{prefix}])")
    return "\n".join(lines)


def fmt_int_array(arr, name, indent=4, per_line=10):
    """Format an integer array."""
    prefix = " " * indent
    vals = arr.tolist()
    if len(vals) <= per_line:
        val_str = ", ".join(str(int(v)) for v in vals)
        return f"{prefix}{name} = jnp.array([{val_str}], dtype=jnp.int32)"
    lines = []
    lines.append(f"{prefix}{name} = jnp.array([")
    for i in range(0, len(vals), per_line):
        chunk = vals[i:i+per_line]
        chunk_str = ", ".join(str(int(v)) for v in chunk)
        lines.append(f"{prefix}    {chunk_str},")
    lines.append(f"{prefix}], dtype=jnp.int32)")
    return "\n".join(lines)


def _fmt_val(v):
    if isinstance(v, float):
        if np.isinf(v):
            return "float('inf')" if v > 0 else "-float('inf')"
        if np.isnan(v):
            return "float('nan')"
        if v == int(v) and abs(v) < 1e10:
            return f"{v:.1f}"
        return repr(v)
    return repr(v)


def get_col(df, col, default=None):
    """Safely get a column from a DataFrame."""
    if col in df.columns:
        return np.array(df[col].values, dtype=np.float64)
    return default


def has_col(df, col):
    return col in df.columns


def convert_case(case_cls, func_name, case_id_str, description, grid_type):
    """Convert a single case class to a JAX case file string."""
    case = case_cls(mock=False)
    
    nodes = case.nodes
    units = case.units
    lines = case.lines
    loads = case.loads
    
    base_mva = getattr(case, 'baseMVA', 100.0)
    
    # Check if unit/line references buses not in nodes (e.g. Case533mt_hi missing bus 1)
    all_node_ids = set(float(x) for x in nodes['id'].values)
    referenced_ids = set()
    for bid in units['bus_id'].values:
        referenced_ids.add(float(bid))
    for bid in lines['from'].values:
        referenced_ids.add(float(bid))
    for bid in lines['to'].values:
        referenced_ids.add(float(bid))
    missing_ids = sorted(referenced_ids - all_node_ids)
    
    if missing_ids:
        import pandas as pd
        # Add missing buses as PQ type with zero load
        extra_rows = []
        for mid in missing_ids:
            row = {'id': mid}
            if has_col(nodes, 'type'):
                row['type'] = 3.0  # slack for missing substation bus
            for c in nodes.columns:
                if c not in row and c != '#id':
                    row[c] = 0.0
            extra_rows.append(row)
        extra_df = pd.DataFrame(extra_rows)
        for c in nodes.columns:
            if c not in extra_df.columns and c != '#id':
                extra_df[c] = 0.0
        nodes = pd.concat([extra_df[nodes.columns.drop('#id', errors='ignore')], nodes], ignore_index=True)
        # Sort by id
        nodes = nodes.sort_values('id').reset_index(drop=True)
    
    n_nodes = len(nodes)
    n_units = len(units)
    n_lines = len(lines)
    n_loads = len(loads)
    
    # Determine slack bus
    if has_col(nodes, 'type'):
        types = np.array(nodes['type'].values, dtype=np.float64)
        slack_mask = types == 3
        if slack_mask.any():
            slack_bus_id = float(nodes['id'].values[np.argmax(slack_mask)])
        else:
            slack_bus_id = float(nodes['id'].values[0])
    else:
        slack_bus_id = float(nodes['id'].values[0])
    
    is_three_phase = hasattr(case, 'PHASE') and str(getattr(case, 'PHASE', '1')) == '3'
    has_ac_lines = has_col(lines, 'r')
    has_uc = has_col(units, 'ramp_up') or has_col(units, 'min_up_time')
    has_fuel = has_col(units, 'type')
    
    parts = []
    
    # Header
    parts.append(f'"""')
    parts.append(f'{case_id_str}: {description}')
    parts.append(f'')
    parts.append(f'Auto-generated from PowerZoo {case_cls.__name__}.')
    parts.append(f'Grid type: {grid_type}, {n_nodes} buses, {n_units} generators, {n_lines} branches, {n_loads} loads')
    parts.append(f'Base MVA: {base_mva}')
    parts.append(f'"""')
    parts.append(f'')
    parts.append(f'import jax.numpy as jnp')
    parts.append(f'from powerzoojax.case.case_data import CaseData')
    parts.append(f'from powerzoojax.case.case_matrices import build_case_matrices')
    parts.append(f'')
    parts.append(f'')
    parts.append(f'def {func_name}() -> CaseData:')
    parts.append(f'    """Create the {case_id_str} case."""')
    parts.append(f'')
    
    # Node data
    parts.append(f'    # ========== Node Data ==========')
    node_ids = np.array(nodes['id'].values, dtype=np.float64)
    parts.append(fmt_array(node_ids, 'node_ids'))
    
    if has_col(nodes, 'type'):
        parts.append(fmt_int_array(np.array(nodes['type'].values, dtype=np.float64), 'node_type'))
    else:
        arr = np.full(n_nodes, 1)
        arr[0] = 3  # first bus is slack
        parts.append(fmt_int_array(arr, 'node_type'))
    
    if is_three_phase:
        for phase, suffix in [('A', 'a'), ('B', 'b'), ('C', 'c')]:
            pd_col = f'Pd_{phase}'
            qd_col = f'Qd_{phase}'
            pd = get_col(nodes, pd_col, np.zeros(n_nodes))
            qd = get_col(nodes, qd_col, np.zeros(n_nodes))
            parts.append(fmt_array(pd, f'node_pd_{suffix}'))
            parts.append(fmt_array(qd, f'node_qd_{suffix}'))
        # Total Pd/Qd for backward compat
        pd_total = (get_col(nodes, 'Pd_A', np.zeros(n_nodes)) +
                     get_col(nodes, 'Pd_B', np.zeros(n_nodes)) +
                     get_col(nodes, 'Pd_C', np.zeros(n_nodes)))
        qd_total = (get_col(nodes, 'Qd_A', np.zeros(n_nodes)) +
                     get_col(nodes, 'Qd_B', np.zeros(n_nodes)) +
                     get_col(nodes, 'Qd_C', np.zeros(n_nodes)))
        parts.append(fmt_array(pd_total, 'node_pd'))
        parts.append(fmt_array(qd_total, 'node_qd'))
    else:
        node_pd = get_col(nodes, 'Pd', np.zeros(n_nodes))
        node_qd = get_col(nodes, 'Qd', np.zeros(n_nodes))
        parts.append(fmt_array(node_pd, 'node_pd'))
        parts.append(fmt_array(node_qd, 'node_qd'))
    
    node_gs = get_col(nodes, 'Gs', np.zeros(n_nodes))
    node_bs = get_col(nodes, 'Bs', np.zeros(n_nodes))
    parts.append(fmt_array(node_gs, 'node_gs'))
    parts.append(fmt_array(node_bs, 'node_bs'))
    
    if has_col(nodes, 'Vmin'):
        parts.append(fmt_array(get_col(nodes, 'Vmin'), 'node_v_min'))
        parts.append(fmt_array(get_col(nodes, 'Vmax'), 'node_v_max'))
    else:
        parts.append(f'    node_v_min = jnp.full({n_nodes}, 0.94)')
        parts.append(f'    node_v_max = jnp.full({n_nodes}, 1.06)')
    
    if has_col(nodes, 'x'):
        parts.append(fmt_array(get_col(nodes, 'x'), 'node_x'))
        parts.append(fmt_array(get_col(nodes, 'y'), 'node_y'))
    else:
        parts.append(f'    node_x = jnp.zeros({n_nodes})')
        parts.append(f'    node_y = jnp.zeros({n_nodes})')
    
    parts.append(f'')
    
    # Unit data
    parts.append(f'    # ========== Unit/Generator Data ==========')
    unit_ids = np.array(units['id'].values, dtype=np.float64)
    unit_bus_ids = np.array(units['bus_id'].values, dtype=np.float64)
    parts.append(fmt_array(unit_ids, 'unit_ids'))
    parts.append(fmt_array(unit_bus_ids, 'unit_bus_ids'))
    
    # Cost coefficients
    parts.append(fmt_array(get_col(units, 'mc_a', np.zeros(n_units)), 'unit_cost_a'))
    parts.append(fmt_array(get_col(units, 'mc_b', np.zeros(n_units)), 'unit_cost_b'))
    parts.append(fmt_array(get_col(units, 'mc_c', np.zeros(n_units)), 'unit_cost_c'))
    
    # Power limits
    if has_col(units, 'p_max'):
        parts.append(fmt_array(get_col(units, 'p_max'), 'unit_p_max'))
        parts.append(fmt_array(get_col(units, 'p_min'), 'unit_p_min'))
    elif has_col(units, 'Pmax'):
        parts.append(fmt_array(get_col(units, 'Pmax'), 'unit_p_max'))
        parts.append(fmt_array(get_col(units, 'Pmin'), 'unit_p_min'))
    else:
        parts.append(f'    unit_p_max = jnp.zeros({n_units})')
        parts.append(f'    unit_p_min = jnp.zeros({n_units})')
    
    # AC unit fields
    if has_col(units, 'Pg'):
        parts.append(fmt_array(get_col(units, 'Pg'), 'unit_pg'))
    else:
        parts.append(f'    unit_pg = jnp.zeros({n_units})')
    if has_col(units, 'Qg'):
        parts.append(fmt_array(get_col(units, 'Qg'), 'unit_qg'))
    else:
        parts.append(f'    unit_qg = jnp.zeros({n_units})')
    if has_col(units, 'Qmax'):
        parts.append(fmt_array(get_col(units, 'Qmax'), 'unit_q_max'))
        parts.append(fmt_array(get_col(units, 'Qmin'), 'unit_q_min'))
    else:
        parts.append(f'    unit_q_max = jnp.zeros({n_units})')
        parts.append(f'    unit_q_min = jnp.zeros({n_units})')
    if has_col(units, 'Vg'):
        parts.append(fmt_array(get_col(units, 'Vg'), 'unit_vg'))
    else:
        parts.append(f'    unit_vg = jnp.ones({n_units})')
    if has_col(units, 'mBase'):
        parts.append(fmt_array(get_col(units, 'mBase'), 'unit_mbase'))
    else:
        parts.append(f'    unit_mbase = jnp.full({n_units}, {base_mva})')
    if has_col(units, 'status'):
        parts.append(fmt_array(get_col(units, 'status'), 'unit_status'))
    else:
        parts.append(f'    unit_status = jnp.ones({n_units})')
    
    # UC fields
    if has_uc or has_fuel:
        parts.append(f'')
        parts.append(f'    # ========== UC Fields ==========')
        if has_col(units, 'ramp_up'):
            parts.append(fmt_array(get_col(units, 'ramp_up'), 'unit_ramp_up'))
        if has_col(units, 'ramp_down'):
            parts.append(fmt_array(get_col(units, 'ramp_down'), 'unit_ramp_down'))
        if has_col(units, 'min_up_time'):
            parts.append(fmt_array(get_col(units, 'min_up_time'), 'unit_min_up_time'))
        if has_col(units, 'min_down_time'):
            parts.append(fmt_array(get_col(units, 'min_down_time'), 'unit_min_down_time'))
        if has_col(units, 'init_power'):
            parts.append(fmt_array(get_col(units, 'init_power'), 'unit_init_power'))
        if has_col(units, 'init_state'):
            parts.append(fmt_array(get_col(units, 'init_state'), 'unit_init_state'))
        if has_col(units, 'init_start_up_cost'):
            parts.append(fmt_array(get_col(units, 'init_start_up_cost'), 'unit_startup_cost'))
        if has_col(units, 'init_no_load_cost'):
            parts.append(fmt_array(get_col(units, 'init_no_load_cost'), 'unit_no_load_cost'))
        if has_col(units, 'keep_time'):
            parts.append(fmt_array(get_col(units, 'keep_time'), 'unit_keep_time'))
        if has_fuel:
            fuel_types = units['type'].values
            fuel_int = np.array([FUEL_MAP.get(str(f).strip().lower(), 0) for f in fuel_types], dtype=np.float64)
            parts.append(fmt_int_array(fuel_int, 'unit_fuel_type'))
    
    parts.append(f'')
    
    # Line data
    parts.append(f'    # ========== Line/Branch Data ==========')
    line_ids = np.array(lines['id'].values, dtype=np.float64)
    line_from_arr = np.array(lines['from'].values, dtype=np.float64)
    line_to_arr = np.array(lines['to'].values, dtype=np.float64)
    
    # Compute line_x: standard cases have 'x' column; three-phase cases
    # derive impedance from line_config * length
    if has_col(lines, 'x'):
        line_x_arr = np.array(lines['x'].values, dtype=np.float64)
    elif hasattr(case, 'line_config') and has_col(lines, 'config_name'):
        lc = case.line_config
        z_cols = ['Z11', 'Z22', 'Z33']
        z_diag = {}
        for cid in lc['id'].values:
            idx = lc.index[lc['id'] == cid][0]
            z_diag[int(cid)] = [lc.at[idx, c] for c in z_cols]
        configs = lines['config_name'].values
        lengths = np.array(lines['length'].values, dtype=np.float64)
        line_r_arr = np.zeros(n_lines)
        line_x_arr = np.zeros(n_lines)
        for i, (cn, ln) in enumerate(zip(configs, lengths)):
            zs = z_diag.get(int(cn), [0j, 0j, 0j])
            # Use the max non-zero diagonal impedance as representative
            best = max(zs, key=lambda z: abs(z))
            line_r_arr[i] = np.real(best) * ln
            line_x_arr[i] = np.imag(best) * ln
        # Replace zeros with small epsilon for PTDF computation
        line_x_arr = np.where(line_x_arr > 0, line_x_arr, 1e-4)
    else:
        line_x_arr = np.ones(n_lines) * 0.01  # fallback
    
    parts.append(fmt_array(line_ids, 'line_ids'))
    parts.append(fmt_array(line_from_arr, 'line_from'))
    parts.append(fmt_array(line_to_arr, 'line_to'))
    parts.append(fmt_array(line_x_arr, 'line_x'))
    
    # Floor/cap with ClearCase convention (0 -> +-1e6)
    floor_arr = np.array(lines['floor'].values, dtype=np.float64)
    cap_arr = np.array(lines['cap'].values, dtype=np.float64)
    floor_arr[floor_arr == 0] = -1e6
    cap_arr[cap_arr == 0] = 1e6
    parts.append(fmt_array(floor_arr, 'line_floor'))
    parts.append(fmt_array(cap_arr, 'line_cap'))
    
    # AC line fields
    has_3ph_impedance = hasattr(case, 'line_config') and has_col(lines, 'config_name')
    if has_ac_lines:
        parts.append(fmt_array(get_col(lines, 'r', np.zeros(n_lines)), 'line_r'))
        parts.append(fmt_array(get_col(lines, 'b', np.zeros(n_lines)), 'line_b'))
        parts.append(fmt_array(get_col(lines, 'ratio', np.zeros(n_lines)), 'line_ratio'))
        parts.append(fmt_array(get_col(lines, 'angle', np.zeros(n_lines)), 'line_angle'))
        parts.append(fmt_array(get_col(lines, 'status', np.ones(n_lines)), 'line_status'))
        if has_col(lines, 'rateA'):
            parts.append(fmt_array(get_col(lines, 'rateA'), 'line_rate_a'))
    elif has_3ph_impedance:
        # line_r was already computed above from config
        parts.append(fmt_array(line_r_arr, 'line_r'))
        parts.append(f'    line_b = jnp.zeros({n_lines})')
        parts.append(fmt_array(get_col(lines, 'ratio', np.zeros(n_lines)), 'line_ratio'))
        parts.append(fmt_array(get_col(lines, 'angle', np.zeros(n_lines)), 'line_angle'))
        parts.append(fmt_array(get_col(lines, 'status', np.ones(n_lines)), 'line_status'))
        if has_col(lines, 'rateA'):
            parts.append(fmt_array(get_col(lines, 'rateA'), 'line_rate_a'))
    else:
        parts.append(f'    line_r = jnp.zeros({n_lines})')
        parts.append(f'    line_b = jnp.zeros({n_lines})')
        parts.append(f'    line_ratio = jnp.zeros({n_lines})')
        parts.append(f'    line_angle = jnp.zeros({n_lines})')
        parts.append(f'    line_status = jnp.ones({n_lines})')
    
    parts.append(f'')
    
    # Load data
    parts.append(f'    # ========== Load Data ==========')
    load_ids = np.array(loads['id'].values, dtype=np.float64)
    load_bus_ids = np.array(loads['bus_id'].values, dtype=np.float64)
    parts.append(fmt_array(load_ids, 'load_ids'))
    parts.append(fmt_array(load_bus_ids, 'load_bus_ids'))
    parts.append(fmt_array(get_col(loads, 'd_max', np.zeros(n_loads)), 'load_d_max'))
    parts.append(fmt_array(get_col(loads, 'd_min', np.zeros(n_loads)), 'load_d_min'))
    
    parts.append(f'')
    
    # Build matrices
    parts.append(f'    # ========== Compute DC Matrices ==========')
    parts.append(f'    matrices = build_case_matrices(')
    parts.append(f'        node_ids=node_ids,')
    parts.append(f'        unit_bus_ids=unit_bus_ids,')
    parts.append(f'        load_bus_ids=load_bus_ids,')
    parts.append(f'        line_from=line_from,')
    parts.append(f'        line_to=line_to,')
    parts.append(f'        line_x=line_x,')
    parts.append(f'        slack_bus_id={slack_bus_id},')
    parts.append(f'    )')
    parts.append(f'')
    
    # Return CaseData
    parts.append(f'    return CaseData(')
    parts.append(f'        n_nodes={n_nodes}, n_lines={n_lines}, n_units={n_units}, n_loads={n_loads},')
    parts.append(f'        node_ids=node_ids, node_x=node_x, node_y=node_y,')
    parts.append(f'        unit_ids=unit_ids, unit_bus_ids=unit_bus_ids,')
    parts.append(f'        unit_p_min=unit_p_min, unit_p_max=unit_p_max,')
    parts.append(f'        unit_cost_a=unit_cost_a, unit_cost_b=unit_cost_b, unit_cost_c=unit_cost_c,')
    parts.append(f'        line_ids=line_ids, line_from=line_from, line_to=line_to,')
    parts.append(f'        line_x=line_x, line_cap=line_cap, line_floor=line_floor,')
    parts.append(f'        load_ids=load_ids, load_bus_ids=load_bus_ids,')
    parts.append(f'        load_d_max=load_d_max, load_d_min=load_d_min,')
    parts.append(f"        PTDF=matrices['PTDF'],")
    parts.append(f"        nodes_units_map=matrices['nodes_units_map'],")
    parts.append(f"        nodes_loads_map=matrices['nodes_loads_map'],")
    parts.append(f"        unit_node_idx=matrices['unit_node_idx'],")
    parts.append(f"        load_node_idx=matrices['load_node_idx'],")
    parts.append(f"        line_from_idx=matrices['line_from_idx'],")
    parts.append(f"        line_to_idx=matrices['line_to_idx'],")
    parts.append(f"        slack_bus_idx=matrices['slack_bus_idx'],")
    
    # AC fields
    parts.append(f'        line_r=line_r, line_b=line_b,')
    parts.append(f'        line_ratio=line_ratio, line_angle=line_angle, line_status=line_status,')
    if has_col(lines, 'rateA'):
        parts.append(f'        line_rate_a=line_rate_a,')
    parts.append(f'        unit_q_min=unit_q_min, unit_q_max=unit_q_max,')
    parts.append(f'        unit_pg=unit_pg, unit_qg=unit_qg, unit_vg=unit_vg,')
    parts.append(f'        unit_status=unit_status, unit_mbase=unit_mbase,')
    parts.append(f'        node_type=node_type, node_pd=node_pd, node_qd=node_qd,')
    parts.append(f'        node_gs=node_gs, node_bs=node_bs,')
    parts.append(f'        node_v_min=node_v_min, node_v_max=node_v_max,')
    parts.append(f'        base_mva={base_mva},')
    
    # UC fields
    if has_uc or has_fuel:
        if has_col(units, 'ramp_up'):
            parts.append(f'        unit_ramp_up=unit_ramp_up, unit_ramp_down=unit_ramp_down,')
        if has_col(units, 'min_up_time'):
            parts.append(f'        unit_min_up_time=unit_min_up_time, unit_min_down_time=unit_min_down_time,')
        if has_col(units, 'init_power'):
            parts.append(f'        unit_init_power=unit_init_power, unit_init_state=unit_init_state,')
        if has_col(units, 'init_start_up_cost'):
            parts.append(f'        unit_startup_cost=unit_startup_cost,')
        if has_col(units, 'init_no_load_cost'):
            parts.append(f'        unit_no_load_cost=unit_no_load_cost,')
        if has_col(units, 'keep_time'):
            parts.append(f'        unit_keep_time=unit_keep_time,')
        if has_fuel:
            parts.append(f'        unit_fuel_type=unit_fuel_type,')
    
    # Three-phase fields
    if is_three_phase:
        parts.append(f'        node_pd_a=node_pd_a, node_qd_a=node_qd_a,')
        parts.append(f'        node_pd_b=node_pd_b, node_qd_b=node_qd_b,')
        parts.append(f'        node_pd_c=node_pd_c, node_qd_c=node_qd_c,')
    
    parts.append(f'    )')
    
    return "\n".join(parts) + "\n"


# ---- Case definitions to convert ----
CASES = [
    # (module_path, class_name, func_name, case_id_str, description, grid_type, output_filename)
    ('powerzoo.case.transmission.Case14', 'Case14', 'create_case14', 'Case14', 'IEEE 14-Bus Test System', 'transmission', 'case14.py'),
    ('powerzoo.case.transmission.Case118', 'Case118', 'create_case118', 'Case118', 'IEEE 118-Bus Test System', 'transmission', 'case118.py'),
    ('powerzoo.case.transmission.Case300', 'Case300', 'create_case300', 'Case300', 'IEEE 300-Bus Test System', 'transmission', 'case300.py'),
    ('powerzoo.case.transmission.Case29GB', 'Case29GB', 'create_case29gb', 'Case29GB', 'GB Reduced 29-Bus Network', 'transmission', 'case29gb.py'),
    ('powerzoo.case.transmission.Case552GB', 'Case552GB', 'create_case552gb', 'Case552GB', 'GB 552-bus transmission', 'transmission', 'case552gb.py'),
    ('powerzoo.case.transmission.Case1354pegase', 'Case1354pegase', 'create_case1354pegase', 'Case1354pegase', 'PEGASE 1354-Bus System', 'transmission', 'case1354pegase.py'),
    ('powerzoo.case.transmission.Case2383wp', 'Case2383wp', 'create_case2383wp', 'Case2383wp', 'Polish 2383-Bus Winter Peak', 'transmission', 'case2383wp.py'),
    ('powerzoo.case.distribution.Case118zh', 'Case118zh', 'create_case118zh', 'Case118zh', '118-Bus Distribution System', 'distribution', 'case118zh.py'),
    ('powerzoo.case.distribution.Case123', 'Case123', 'create_case123', 'Case123', 'IEEE 123-Bus Three-Phase Distribution', 'distribution', 'case123.py'),
    ('powerzoo.case.distribution.Case141', 'Case141', 'create_case141', 'Case141', '141-Bus Distribution System', 'distribution', 'case141.py'),
    ('powerzoo.case.distribution.Case533mt_hi', 'Case533mt_hi', 'create_case533mt_hi', 'Case533mt_hi', '533-Bus Medium Tension (Hi)', 'distribution', 'case533mt_hi.py'),
    ('powerzoo.case.distribution.Case533mt_lo', 'Case533mt_lo', 'create_case533mt_lo', 'Case533mt_lo', '533-Bus Medium Tension (Lo)', 'distribution', 'case533mt_lo.py'),
]


def main():
    import importlib
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for module_path, class_name, func_name, case_id_str, description, grid_type, output_filename in CASES:
        print(f"Converting {case_id_str}...")
        try:
            module = importlib.import_module(module_path)
            case_cls = getattr(module, class_name)
            content = convert_case(case_cls, func_name, case_id_str, description, grid_type)
            
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_path, 'w') as f:
                f.write(content)
            print(f"  -> {output_path} ({len(content)} bytes)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
