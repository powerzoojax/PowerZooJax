# Cases

!!! note "Python API 签名"
    本页只翻译概览、合约表与示例。完整的 mkdocstrings 自动生成签名（参数、字段、类型）由英文 API 渲染，见 [English API → Cases](../../en/api/case.md)。

`powerzoojax.case` 包把 benchmark 网络转成 JAX 友好的数值结构。`CaseData` 是会进 JIT 编译代码的对象；人类可读的元数据留在 registry 层。

cases 的架构层角色见 [Architecture → Repo map](../architecture/repo-map.md#powerzoojaxcase)。

## 加载与发现

```python
from powerzoojax.case import load_case, list_cases, get_meta

case = load_case("5")
meta = get_meta("33bw")

for item in list_cases(grid_type="distribution"):
    print(item.name, item.bus_count, item.phase)
```

内置 ID：

- 输电网：`5`、`14`、`118`、`300`、`1354pegase`、`2383wp`、`29gb`、`552gb`。
- 配电网：`33bw`、`118zh`、`123`、`141`、`533mt_hi`、`533mt_lo`。

## `CaseData` 约定

重要字段：

- `PTDF`、`nodes_units_map`、`nodes_loads_map`。
- `unit_p_min`、`unit_p_max`。
- `unit_cost_a`、`unit_cost_b`、`unit_cost_c`。
- `unit_ramp_up`、`unit_ramp_down`、`min_up_time`、`min_down_time`、`unit_startup_cost`、`unit_no_load_cost`（UC 专用；case118 已填充）。
- `line_cap`、`line_floor`。
- AC 专用字段：`line_r`、`line_b`、`node_type`、`node_v_min`、`node_v_max`。
- 三相负荷字段：`node_pd_a`、`node_qd_a`、…

`CaseData` 故意不带字符串 `name` 字段。JAX trace 数值化的 case 对象；名字与描述放在 `CaseMeta` 里。

## API

`load_case`、`list_cases`、`get_meta`、`CaseMeta`、`CaseData` 与 `validate_case_data` 的完整签名由英文 API 页渲染。

## 矩阵 helper

`build_case_matrices`、`compute_ptdf`、邻接矩阵 / 度矩阵 / 拉普拉斯矩阵 helper 的完整签名由英文 API 页渲染。

## 转换与检查

`case_to_jax`、`convert_case`、`CaseInfo` 与 `CasePlotter` 的完整签名由英文 API 页渲染。
