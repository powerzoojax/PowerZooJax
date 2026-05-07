#!/usr/bin/env python3
"""Generate table-format case files from PowerZoo sources.

Reads each PowerZoo Case class and writes a readable, editable Python file
using the 2-D table format + ``build_case_from_tables()``.
"""

import sys, os, importlib, textwrap
import numpy as np

sys.path.insert(0, os.path.expanduser("~/codes/PowerZoo"))

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "..",
                           "powerzoojax", "case", "cases")

FUEL_MAP = {'nuclear': 1, 'coal': 2, 'gas': 3, 'oil': 4,
            'hydro': 5, 'wind': 6, 'solar': 7}

def _fmt(v, width=10):
    """Format a single value for table display."""
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, (float, np.floating)):
        if np.isinf(v):
            return "float('inf')" if v > 0 else "-float('inf')"
        if v == int(v) and abs(v) < 1e10:
            return f"{v:g}"
        return f"{v:g}"
    return str(v)


def _format_table(cols, data, indent=4):
    """Format columns + data as Python source."""
    prefix = " " * indent
    lines = []
    # Column header
    col_str = ", ".join(f'"{c}"' for c in cols)
    lines.append(f"{prefix}[{col_str}],")
    # Data rows
    for row in data:
        vals = ", ".join(_fmt(v) for v in row)
        lines.append(f"{prefix}[{vals}],")
    return "\n".join(lines)


def convert_case(case_cls, func_name, case_id, desc, grid_type, subdir):
    case = case_cls(mock=False)
    base_mva = getattr(case, 'baseMVA', 100.0)

    def clean_cols(df):
        return [c for c in df.columns if c != '#id']

    def clean_data(df, cols):
        return df[cols].values.tolist()

    # Handle three-phase lines
    lines_df = case.lines
    if hasattr(case, 'line_config') and 'config_name' in case.lines.columns:
        from powerzoojax.case.case_builder import _compute_3ph_impedance
        lines_df = _compute_3ph_impedance(case)

    # Handle missing buses
    nodes_df = case.nodes
    from powerzoojax.case.case_builder import _ensure_referenced_buses
    nodes_df = _ensure_referenced_buses(case)

    n_cols = clean_cols(nodes_df)
    u_cols = clean_cols(case.units)
    l_cols = clean_cols(lines_df)
    d_cols = clean_cols(case.loads)

    n_data = clean_data(nodes_df, n_cols)
    u_data = clean_data(case.units, u_cols)
    l_data = clean_data(lines_df, l_cols)
    d_data = clean_data(case.loads, d_cols)

    src = []
    src.append(f'"""{case_id}: {desc}')
    src.append(f'')
    src.append(f'Grid type: {grid_type}')
    n_nodes, n_units, n_lines, n_loads = len(n_data), len(u_data), len(l_data), len(d_data)
    src.append(f'{n_nodes} buses, {n_units} generators, {n_lines} branches, {n_loads} loads')
    src.append(f'Base MVA: {base_mva}')
    src.append(f'"""')
    src.append(f'')
    src.append(f'from powerzoojax.case.case_builder import build_case_from_tables')
    src.append(f'from powerzoojax.case.case_data import CaseData')
    src.append(f'')
    src.append(f'')
    src.append(f'def {func_name}() -> CaseData:')
    src.append(f'    """Create {case_id} ({n_nodes} buses, {grid_type})."""')
    src.append(f'')
    src.append(f'    _nodes = [')
    src.append(_format_table(n_cols, n_data, 8))
    src.append(f'    ]')
    src.append(f'')
    src.append(f'    _units = [')
    src.append(_format_table(u_cols, u_data, 8))
    src.append(f'    ]')
    src.append(f'')
    src.append(f'    _lines = [')
    src.append(_format_table(l_cols, l_data, 8))
    src.append(f'    ]')
    src.append(f'')
    src.append(f'    _loads = [')
    src.append(_format_table(d_cols, d_data, 8))
    src.append(f'    ]')
    src.append(f'')
    src.append(f'    return build_case_from_tables(')
    src.append(f'        nodes_cols=_nodes[0], nodes_data=_nodes[1:],')
    src.append(f'        units_cols=_units[0], units_data=_units[1:],')
    src.append(f'        lines_cols=_lines[0], lines_data=_lines[1:],')
    src.append(f'        loads_cols=_loads[0], loads_data=_loads[1:],')
    src.append(f'        base_mva={base_mva},')
    src.append(f'    )')
    src.append(f'')

    return "\n".join(src)


CASES = [
    # (module_path, class_name, func_name, case_id, desc, grid_type, subdir, filename)
    ('powerzoo.case.transmission.Case5', 'Case5', 'create_case5', 'Case5', 'IEEE 5-Bus Test System', 'transmission', 'transmission', 'case5.py'),
    ('powerzoo.case.transmission.Case14', 'Case14', 'create_case14', 'Case14', 'IEEE 14-Bus Test System', 'transmission', 'transmission', 'case14.py'),
    ('powerzoo.case.transmission.Case118', 'Case118', 'create_case118', 'Case118', 'IEEE 118-Bus Test System', 'transmission', 'transmission', 'case118.py'),
    ('powerzoo.case.transmission.Case300', 'Case300', 'create_case300', 'Case300', 'IEEE 300-Bus Test System', 'transmission', 'transmission', 'case300.py'),
    ('powerzoo.case.transmission.Case29GB', 'Case29GB', 'create_case29gb', 'Case29GB', 'GB Reduced 29-Bus Network', 'transmission', 'transmission', 'case29gb.py'),
    ('powerzoo.case.transmission.Case552GB', 'Case552GB', 'create_case552gb', 'Case552GB', 'GB 552-bus transmission', 'transmission', 'transmission', 'case552gb.py'),
    ('powerzoo.case.transmission.Case1354pegase', 'Case1354pegase', 'create_case1354pegase', 'Case1354pegase', 'PEGASE 1354-Bus System', 'transmission', 'transmission', 'case1354pegase.py'),
    ('powerzoo.case.transmission.Case2383wp', 'Case2383wp', 'create_case2383wp', 'Case2383wp', 'Polish 2383-Bus Winter Peak', 'transmission', 'transmission', 'case2383wp.py'),
    ('powerzoo.case.distribution.Case33bw', 'Case33bw', 'create_case33bw', 'Case33bw', 'IEEE 33-Bus Distribution', 'distribution', 'distribution', 'case33bw.py'),
    ('powerzoo.case.distribution.Case118zh', 'Case118zh', 'create_case118zh', 'Case118zh', '118-Bus Distribution', 'distribution', 'distribution', 'case118zh.py'),
    ('powerzoo.case.distribution.Case123', 'Case123', 'create_case123', 'Case123', 'IEEE 123-Bus Three-Phase Distribution', 'distribution', 'distribution', 'case123.py'),
    ('powerzoo.case.distribution.Case141', 'Case141', 'create_case141', 'Case141', '141-Bus Distribution', 'distribution', 'distribution', 'case141.py'),
    ('powerzoo.case.distribution.Case533mt_hi', 'Case533mt_hi', 'create_case533mt_hi', 'Case533mt_hi', '533-Bus Medium Tension (Hi)', 'distribution', 'distribution', 'case533mt_hi.py'),
    ('powerzoo.case.distribution.Case533mt_lo', 'Case533mt_lo', 'create_case533mt_lo', 'Case533mt_lo', '533-Bus Medium Tension (Lo)', 'distribution', 'distribution', 'case533mt_lo.py'),
]


def main():
    for mod_path, cls_name, func_name, case_id, desc, grid_type, subdir, filename in CASES:
        print(f"Generating {case_id}...")
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            content = convert_case(cls, func_name, case_id, desc, grid_type, subdir)

            out_dir = os.path.join(OUTPUT_BASE, subdir)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, filename)
            with open(out_path, 'w') as f:
                f.write(content)
            print(f"  -> {out_path} ({len(content)} bytes)")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
