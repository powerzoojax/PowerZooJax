# Cases

The `powerzoojax.case` package converts benchmark networks into JAX-friendly numerical structures. `CaseData` is the object that crosses into JIT-compiled code; human-readable metadata stays in the registry layer.

For the architecture-level role of cases, see [Architecture → Repo map](../architecture/repo-map.md#powerzoojaxcase).

## Loading and discovery

```python
from powerzoojax.case import load_case, list_cases, get_meta

case = load_case("5")
meta = get_meta("33bw")

for item in list_cases(grid_type="distribution"):
    print(item.name, item.bus_count, item.phase)
```

Built-in IDs:

- transmission: `5`, `14`, `118`, `300`, `1354pegase`, `2383wp`, `29gb`, `552gb`.
- distribution: `33bw`, `118zh`, `123`, `141`, `533mt_hi`, `533mt_lo`.

## `CaseData` conventions

Important fields:

- `PTDF`, `nodes_units_map`, `nodes_loads_map`.
- `unit_p_min`, `unit_p_max`.
- `unit_cost_a`, `unit_cost_b`, `unit_cost_c`.
- `unit_ramp_up`, `unit_ramp_down`, `min_up_time`, `min_down_time`, `unit_startup_cost`, `unit_no_load_cost` (UC-specific; populated for case118).
- `line_cap`, `line_floor`.
- AC-only fields: `line_r`, `line_b`, `node_type`, `node_v_min`, `node_v_max`.
- three-phase load fields: `node_pd_a`, `node_qd_a`, ...

`CaseData` intentionally has no string `name` field. JAX traces the numeric case object; names and descriptions live in `CaseMeta`.

## API

::: powerzoojax.case.load_case

::: powerzoojax.case._registry.list_cases

::: powerzoojax.case._registry.get_meta

::: powerzoojax.case._registry.CaseMeta

::: powerzoojax.case.case_data.CaseData

::: powerzoojax.case.case_data.validate_case_data

## Matrix helpers

::: powerzoojax.case.case_matrices.build_case_matrices

::: powerzoojax.case.case_matrices.compute_ptdf

::: powerzoojax.case.case_matrices.compute_adjacency_matrix

::: powerzoojax.case.case_matrices.compute_degree_matrix

::: powerzoojax.case.case_matrices.compute_laplacian_matrix

## Conversion and inspection

::: powerzoojax.case.case_adapter.case_to_jax

::: powerzoojax.case.case_adapter.convert_case

::: powerzoojax.case.case_info.CaseInfo

::: powerzoojax.case.case_plotter.CasePlotter
