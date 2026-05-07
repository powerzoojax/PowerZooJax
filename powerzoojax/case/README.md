# PowerZooJAX Case Module

## Directory Layout

```
powerzoojax/case/
├── raw_cases/          # Original OOP-style cases
│   ├── case5.py        # 5-bus case
│   ├── case33bw.py     # 33-bus case
│   └── case118.py      # 118-bus case
│
├── cases/              # Native JAX cases with flat arrays
│   ├── case5.py        # Flat implementation from scratch
│   └── case33bw.py     # Flat implementation from scratch
│
├── dataframe.py        # DataFrame class used by raw_cases
├── case_adapter.py     # Adapter from raw cases to JAX
├── case_data.py        # CaseData definition
└── case_matrices.py    # Matrix computations
```

## Two Equivalent Paths

### Path 1: Raw Cases + Adapter (recommended)

**Advantages**: readable and easy to maintain.

```python
from powerzoojax.case.raw_cases import Case5
from powerzoojax.case import case_to_jax

# 1. Create the original OOP-style case
case_raw = Case5()

# 2. Convert it to the JAX format
case_data = case_to_jax(case_raw)

# 3. Use it for training
@jax.jit
def train_step(case_data, state):
    return case_data.PTDF @ state.injections
```

### Path 2: Native JAX Cases (alternative)

**Advantages**: pure JAX with no adapter layer.

```python
from powerzoojax.case import create_case5

# Create the JAX format directly
case_data = create_case5()
```

## Design Rationale

| Property | Raw Cases | Native Cases |
|------|-----------|--------------|
| **Style** | OOP + DataFrame | Flat arrays |
| **Definition** | Readable | Explicit and efficient |
| **Maintenance** | Easy to modify | Requires more care |
| **Use case** | Recommended for routine use | Alternative/reference implementation |
| **Result** | Equivalent | Equivalent |

## Adding a New Case

### Add to raw_cases/ (recommended)

1. Create `raw_cases/caseXX.py`:

```python
from powerzoojax.case.dataframe import DataFrame

class CaseXX:
    def __init__(self):
        self.nodes = DataFrame(
            ['id', 'x', 'y'],
            [[1.0, 0, 0], ...]
        )
        
        self.units = DataFrame(
            ['id', 'bus_id', ...],
            [[1.0, 1.0, ...], ...]
        )
        
        # ... lines, loads
        
        # Fix zero limits
        self.lines.loc[self.lines['cap'] == 0, 'cap'] = 1000000
        self.lines.loc[self.lines['floor'] == 0, 'floor'] = -1000000
        
        self.name = 'CaseXX'
```

2. Export it in `raw_cases/__init__.py`:

```python
from powerzoojax.case.raw_cases.caseXX import CaseXX
__all__ = [..., 'CaseXX']
```

3. Use it:

```python
from powerzoojax.case.raw_cases import CaseXX
from powerzoojax.case import case_to_jax
case_data = case_to_jax(CaseXX())
```

### Add to cases/ (optional)

If a pure JAX reference implementation is needed, add a flat version under `cases/`.

## Verifying Equivalence

Run the equivalence test:

```bash
python examples/case_equivalence_test.py
```

## Key Points

1. **raw_cases/** has no dependency on the external PowerZoo package.
2. **DataFrame** is a lightweight local implementation.
3. **The adapter** runs once on the CPU and does not affect training performance.
4. **Both paths** produce equivalent `CaseData`.
5. **The recommended path** is raw cases plus the adapter.
