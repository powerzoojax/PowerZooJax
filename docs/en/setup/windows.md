# Running on Windows

This page is a linear happy path for Windows users: from a fresh machine to opening a PowerZooJax checkout living inside WSL Ubuntu in VS Code, with the `uv`-managed `.venv` selected as the Python environment and JAX seeing the GPU. Each section is one step; do them in order.

If you are not on Windows, skip this page and follow the standard flow in [Getting Started](../getting-started.md).

## Why WSL2

JAX's CUDA support is **Linux-first**. The official `jax[cuda12]` wheels only target Linux x86\_64; native Windows JAX is CPU-only. The standard way to get GPU acceleration on Windows is **WSL2 + Ubuntu**: a lightweight Linux subsystem inside Windows that gets near-native CUDA performance through NVIDIA's GPU passthrough. You keep your Windows desktop, browser, and Office apps; PowerZooJax runs on essentially the same runtime as a Linux server.

## Prerequisites

- Windows 11, or Windows 10 21H2 and newer
- An NVIDIA GPU with the latest **Windows-side** Game Ready or Studio Driver (do not install a Linux NVIDIA driver inside WSL)
- 30 GB+ free disk (about 10 GB for the WSL distro, 10 GB for CUDA wheels and build caches, plus headroom for cases and checkpoints)
- VS Code installed on the **Windows side** (not inside WSL)
- Optional but nice: Windows Terminal for a better Ubuntu shell

## Step 1: enable WSL2 and install Ubuntu

In **PowerShell (admin)**:

```powershell
wsl --install
```

This installs the WSL2 kernel and the default Ubuntu distro in one shot. Reboot when it tells you to.

After the reboot, Ubuntu launches automatically and asks for an initial Linux username and password (independent of your Windows account).

Verify:

```powershell
wsl -l -v
```

You should see one row for `Ubuntu` with `VERSION` `2`. If `VERSION` is `1`, convert:

```powershell
wsl --set-version Ubuntu 2
```

## Step 2: verify the NVIDIA GPU inside WSL

In the Ubuntu shell:

```bash
nvidia-smi
```

You should see your GPU name, driver version, and memory usage. If you do not:

1. **Do not install a Linux NVIDIA driver inside WSL.** The Windows-side driver already exposes `libcuda.so` through `/usr/lib/wsl/lib/`; installing a separate Linux driver inside WSL conflicts with it.
2. On the Windows side, install the latest Game Ready / Studio Driver from [nvidia.com](https://www.nvidia.com/Download/index.aspx). CUDA 12 needs driver 525+.
3. From PowerShell, run `wsl --update`, then `wsl --shutdown`, then reopen Ubuntu.

## Step 3: base Ubuntu environment

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential curl git python3.12 python3.12-venv python3.12-dev
```

Install `uv` using the official standalone installer (do not use `pip install uv`):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

If the installer reports a different path (e.g. `~/.cargo/env`), `source` whichever one it prints.

Verify:

```bash
uv --version
python3.12 --version
```

## Step 4: do you need the CUDA Toolkit?

**Short answer: no, do not `apt install nvidia-cuda-toolkit`. PowerZooJax does not need it.**

Running `jax[cuda12]` inside WSL does not require a system-level CUDA toolkit, because:

- **Driver layer** — the Windows NVIDIA driver already exposes `libcuda.so` to WSL through `/usr/lib/wsl/lib/`. This is the only piece that has to come from the host driver.
- **Runtime and cuDNN** — `uv sync --extra cuda12` installs `nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, etc. as pip wheels into `.venv`. Those are the binaries JAX actually links against.
- **`nvcc` compiler** — PowerZooJax is pure JAX. It never compiles custom CUDA C++ kernels, so `nvcc` is not on the critical path.

Installing `nvidia-cuda-toolkit` from `apt` brings in a *different version* of system CUDA libraries alongside the wheels, which usually causes:

- `LD_LIBRARY_PATH` picking the system libs over the wheels, leading to "cuDNN version mismatch" at JAX startup
- The toolkit being upgraded out of sync with the wheels, after which `jax.devices()` reverts to CPU
- Slower WSL startup time

**You only need a system CUDA toolkit if** you are writing your own CUDA kernels and compiling them with `nvcc`, profiling with Nsight Systems / Nsight Compute, or running another framework that explicitly depends on system CUDA. None of those are part of the default PowerZooJax workflow.

To confirm JAX picks up the GPU without the system toolkit, after Step 6 run `uv run python -c "import jax; print(jax.devices())"` and expect `[CudaDevice(id=0)]` — the CUDA runtime that ships in the wheels is doing all the work.

## Step 5: put the repo inside WSL

**Strongly recommended: clone the repo into the WSL filesystem**, e.g. `~/code/PowerZooJax`:

```bash
mkdir -p ~/code
cd ~/code
git clone https://github.com/powerzoojax/PowerZooJax.git
cd PowerZooJax
```

Do **not** keep the repo under `/mnt/c/...` or `/mnt/e/...`. Three reasons:

1. Cross-filesystem IO is 10–100x slower; `uv sync`, `pytest`, and checkpoint writes all suffer.
2. Writing `.egg-info`, `build/`, or `.uv-cache` against the Windows permission model intermittently fails with `Permission denied`.
3. `inotify` file watchers are unreliable over the 9P filesystem, breaking `mkdocs serve` livereload, PyTorch DataLoader workers, and similar.

If your code already lives on the Windows side and you do not want to move it, the workaround is a symlink plus `TMPDIR`:

```bash
ln -s /mnt/<drive>/<path-to>/PowerZooJax ~/PowerZooJax
echo 'export TMPDIR=/tmp' >> ~/.bashrc
source ~/.bashrc
cd ~/PowerZooJax
```

`TMPDIR=/tmp` redirects setuptools build temp files from the repo (on the Windows filesystem) to WSL's own tmpfs, sidestepping the `.egg-info` permission issue. This does not fix the slow livereload — for long-term work, just move the repo into `~/`.

## Step 6: sync dependencies with uv

```bash
cd ~/code/PowerZooJax
uv sync --extra cuda12 --extra rl
```

A few details:

- `--extra cuda12` installs `jax[cuda12]>=0.4.30; sys_platform == 'linux'`. The `sys_platform` marker means it only triggers on Linux (i.e. WSL Ubuntu); native Windows would silently skip it.
- `--extra rl` installs rejax + distrax (single-agent PPO/SAC). For multi-agent, use `--extra marl` (jaxmarl). They are mutually exclusive per `[tool.uv].conflicts` in `pyproject.toml`, so you cannot pass both.
- You can stack additional extras: `--extra benchmarks` (SB3 / sbx-rl), `--extra viz` (networkx), `--extra docs` (full mkdocs stack), `--extra dev` (pytest).
- The first sync pulls roughly 1.5 GB of NVIDIA wheels from PyPI; be patient.

Verify the GPU is visible:

```bash
uv run python -c "import jax; print(jax.devices())"
```

Expected:

```
[CudaDevice(id=0)]
```

If you see `[CpuDevice(id=0)]`, double-check that you are inside WSL Linux, that `--extra cuda12` was passed, and that `uv pip list | grep cuda` shows the `nvidia-*` wheels.

## Step 7: install VS Code and the WSL extension on Windows

VS Code lives on the **Windows side**, not inside WSL (the WSL-side VS Code Server is installed automatically by the extension):

1. Install VS Code from [code.visualstudio.com](https://code.visualstudio.com/).
2. From the Extensions panel, install:
   - `Remote - WSL` (`ms-vscode-remote.remote-wsl`)
   - `WSL` (the renamed pack — same extension)
   - `Python` (`ms-python.python`)
   - `Pylance` (`ms-python.vscode-pylance`)
   - `Ruff` (`charliermarsh.ruff`, optional)

You install these once on the Windows side. When you connect into WSL, VS Code will prompt you to click "Install in WSL: Ubuntu" for each one — that's a one-click action per extension.

## Step 8: open the WSL project in VS Code

Three equivalent entry points; pick whichever feels natural:

**A. From the Ubuntu terminal (fastest)**

```bash
cd ~/code/PowerZooJax
code .
```

The first run installs the WSL-side VS Code Server (~100 MB, one-time); subsequent runs are instant.

**B. From VS Code's command palette**

`Ctrl+Shift+P` → `WSL: Open Folder in WSL...` → pick `Ubuntu` → navigate to `~/code/PowerZooJax`.

**C. From the green status-bar corner**

Click the green `><` icon at the bottom-left of the VS Code window → `Connect to WSL using Distro...` → `Ubuntu` → `Open Folder` → `~/code/PowerZooJax`.

**Verify the connection**: the bottom-left status bar should now say `WSL: Ubuntu`. From Windows Explorer you can also browse WSL files by typing `\\wsl$\Ubuntu\home\<user>\code\PowerZooJax` in the address bar.

For each extension you previously installed on Windows, click "Install in WSL: Ubuntu" once when prompted.

## Step 9: point VS Code at the uv-managed `.venv`

`uv sync` creates `.venv/` at the repo root: `~/code/PowerZooJax/.venv/`, fully inside the WSL filesystem.

To select it as VS Code's interpreter:

1. `Ctrl+Shift+P` → `Python: Select Interpreter`.
2. Pick `.venv/bin/python`. The path looks like `${workspaceFolder}/.venv/bin/python`.

To pin this per-workspace, drop a `.vscode/settings.json` at the repo root:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "files.eol": "\n"
}
```

What each setting does:

- `python.defaultInterpreterPath` — point the default Python at `.venv`; portable across machines and clones.
- `python.terminal.activateEnvironment` — newly opened VS Code terminals auto `source .venv/bin/activate`.
- `files.eol` — keep line endings consistent when you edit the same file from both Windows and WSL.

Verify: open a fresh terminal inside VS Code and run:

```bash
which python
# expected: ~/code/PowerZooJax/.venv/bin/python
python -c "import jax; print(jax.devices())"
# expected: [CudaDevice(id=0)]
```

## Step 10: run a project smoke test

Use the repository examples and benchmark entry points as the public smoke tests:

```bash
uv run python examples/jax_00_verify_device.py
uv run python examples/jax_01_create_case.py
uv run python examples/jax_02_grid_env.py
```

For a direct GPU check:

```python
import jax
import jax.numpy as jnp

print("JAX:", jax.__version__)
print("Devices:", jax.devices())
print("Backend:", jax.default_backend())

x = jnp.arange(1_000_000, dtype=jnp.float32)
y = jax.jit(lambda a: (a * 2).sum())(x).block_until_ready()
print("Sum =", float(y))
```

You should see `Backend: gpu` and `Devices: [CudaDevice(id=0)]`.

The first call that uses a PowerZooJax env may trigger JAX JIT compilation, which can take 30–60 seconds. Subsequent calls are fast.

## Step 11: extra configuration

**WSL resource limits** — drop `%UserProfile%\.wslconfig` (in your Windows user folder, *not* in the project):

```ini
[wsl2]
memory=24GB
processors=8
swap=8GB
```

After saving, `wsl --shutdown` from PowerShell to apply.

**XLA preallocation** — PowerZooJax sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` at import time so JAX does not grab all GPU memory upfront. To override:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=true   # preallocate, more stable scheduling
```

**Browse mkdocs serve from Windows** — just run:

```bash
uv run mkdocs serve
```

Then open `http://localhost:8000` in any Windows browser. WSL2 forwards localhost transparently; no manual port forwarding needed. If livereload misbehaves, `Ctrl+C` and restart `mkdocs serve`.

**Git SSH keys** — generate them inside Ubuntu (`~/.ssh/`), do not reuse the Windows side:

```bash
ssh-keygen -t ed25519 -C "anonymous-key-label"
cat ~/.ssh/id_ed25519.pub  # copy into GitHub
```

**WSL maintenance commands** (PowerShell):

- `wsl --update` — updates the WSL kernel
- `wsl --shutdown` — kills all WSL instances (releases RAM)
- `wsl --status` — version and default distro

## Troubleshooting

| Symptom | What to try |
| --- | --- |
| `uv sync` reports `Permission denied` on `.egg-info` | Move the repo out of `/mnt/...`. If you must keep it there, `export TMPDIR=/tmp` |
| `nvidia-smi: command not found` | Install / update the Windows-side NVIDIA driver; `wsl --update` then `wsl --shutdown` |
| `nvidia-smi` works in WSL but `jax.devices()` only shows `CpuDevice` | `--extra cuda12` was not passed, or current shell is not in `.venv`; `uv pip list \| grep cuda` to confirm wheels |
| JAX raises `Could not load library libcudnn.so` | `--extra cuda12` partially installed: `uv sync --extra cuda12 --reinstall` |
| cuDNN version mismatch error | Earlier `apt install nvidia-cuda-toolkit` introduced system CUDA that conflicts with wheels: `sudo apt remove nvidia-cuda-toolkit`, then `uv sync --extra cuda12` |
| VS Code imports fail, terminal works | Re-select the interpreter with `Python: Select Interpreter` and pick `.venv/bin/python` |
| WSL holds onto memory after work | `wsl --shutdown` from PowerShell; cap with `.wslconfig` |
| `mkdocs serve` livereload stalls or never refreshes | Often because the repo is on `/mnt/...`; move to `~/`, or `Ctrl+C` and restart |
| Browser cannot reach `mkdocs serve` | Use `localhost:8000` rather than `127.0.0.1`; if still stuck, `wsl --shutdown` |
| Garbled CJK file names / paths | `locale` to inspect `LANG`; `sudo dpkg-reconfigure locales` and pick `en_US.UTF-8` or `zh_CN.UTF-8` |
| `jax.devices()` is slow on first run | Normal — CUDA context init; subsequent calls are fast |
| `code .` errors with `code: command not found` | Connect VS Code into WSL once via the green corner; that adds `code` to the WSL `PATH` |

## Next steps

- [Getting Started](../getting-started.md) — first `reset` / `step` and a scan-style rollout
- [Examples](../examples/index.md) — five complete example scripts
- [Benchmarks](../benchmarks/overview.md) — the five paper tasks

For a deeper path through the env layer, continue with [Getting Started](../getting-started.md) and the example scripts.
