# Running on Windows

这一页给 Windows 用户一条线性的 happy path：从空机器到能在 VS Code 里打开 WSL Ubuntu 内的 PowerZooJax 仓库、用 `uv` 管的虚拟环境跑项目示例、并且 GPU 被 JAX 看见。每一节就是一步，按顺序做下来即可。

非 Windows 用户不需要看这一页，按 [Getting Started](../getting-started.md) 的通用流程即可。

## 为什么需要 WSL2

JAX 的 CUDA 支持是 **Linux-first**：官方 `jax[cuda12]` wheel 只发布到 Linux x86\_64，原生 Windows 上的 JAX 只能用 CPU。要在 Windows 上拿到 GPU 加速，标准做法是 **WSL2 + Ubuntu**：在 Windows 内跑一个轻量 Linux 子系统，并通过 NVIDIA 驱动的 GPU 直通获得接近原生的 CUDA 性能。这样既能用 Windows 桌面、Office、浏览器，又能让 PowerZooJax 跑在和 Linux 服务器几乎一致的运行时上。

## 先决条件

- Windows 11，或 Windows 10 21H2 及以上
- 一块 NVIDIA GPU，**Windows 端**装好最新的 Game Ready 或 Studio Driver（不要在 WSL 里装 Linux NVIDIA driver）
- 30 GB 以上空闲磁盘（WSL 发行版约 10 GB，CUDA wheel 与编译缓存约 10 GB，case / checkpoint 留余量）
- VS Code 装在 **Windows 端**（不是 WSL 内）
- 可选：Windows Terminal，比默认 Ubuntu shell 更顺手

## 第一步：启用 WSL2 与 Ubuntu

在 **PowerShell（管理员）** 里：

```powershell
wsl --install
```

这一条命令会同时装 WSL2 内核与默认的 Ubuntu 发行版。装完重启电脑。

重启后 Ubuntu 会自动启动，让你设初始用户名和密码（这是 Linux 用户名，和 Windows 账户独立）。

验证：

```powershell
wsl -l -v
```

应看到一行 `Ubuntu` 且 `VERSION` 是 `2`。如果是 `1`，转 WSL2：

```powershell
wsl --set-version Ubuntu 2
```

## 第二步：在 WSL 内验证 NVIDIA GPU

进入 Ubuntu shell 后跑：

```bash
nvidia-smi
```

应看到 GPU 名称、驱动版本、显存占用。如果没看到：

1. **不要在 WSL 内装 Linux NVIDIA driver**。Windows 端的驱动会通过 `/usr/lib/wsl/lib/` 自动把 `libcuda.so` 暴露给 WSL，在 WSL 内再装一次会冲突。
2. 在 Windows 端从 [nvidia.com](https://www.nvidia.com/Download/index.aspx) 装最新的 Game Ready / Studio Driver，CUDA 12 需要 525 及以上。
3. 在 PowerShell 里跑 `wsl --update`，然后 `wsl --shutdown`，再重新打开 Ubuntu。

## 第三步：Ubuntu 基础环境

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential curl git python3.12 python3.12-venv python3.12-dev
```

装 `uv`（推荐用官方 standalone installer，不要用 `pip install uv`）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

如果安装脚本提示放到了别的路径（比如 `~/.cargo/env`），按它的提示来 `source`。

验证：

```bash
uv --version
python3.12 --version
```

## 第四步：CUDA Toolkit 要不要装？

**短答：不要装 `nvidia-cuda-toolkit`，PowerZooJax 不需要它。**

WSL 里跑 `jax[cuda12]` 不需要系统级 CUDA toolkit，原因如下：

- **driver 部分** —— Windows 端的 NVIDIA driver 已经把 `libcuda.so` 通过 `/usr/lib/wsl/lib/` 直通给 WSL，这是 GPU 真正所需的运行库
- **runtime 与 cuDNN** —— `uv sync --extra cuda12` 会把 `nvidia-cuda-runtime-cu12`、`nvidia-cudnn-cu12`、`nvidia-cublas-cu12` 等 wheel 装进 `.venv`，这是 JAX 真正调用的二进制
- **`nvcc` 编译器** —— PowerZooJax 全部用纯 JAX，不需要写 / 编译自定义 CUDA C++ kernel，因此用不到 `nvcc`

如果你 `apt install nvidia-cuda-toolkit`，会引入一套和 pip wheel 不同版本的系统 CUDA 库，常见后果：

- `LD_LIBRARY_PATH` 把系统库排到 wheel 库前面，JAX 启动时报 cuDNN 版本不匹配
- 升级 toolkit 后 wheel 与 system 不一致，`jax.devices()` 只剩 CPU
- WSL 启动时间变长

**只有以下情况才需要系统 CUDA toolkit**：自己写 CUDA kernel 用 `nvcc` 编译；用 Nsight Systems / Nsight Compute 做 profiling；同机还跑别的明确依赖系统 CUDA 的框架。这些场景都不在 PowerZooJax 默认工作流里。

验证 JAX 不需要系统 toolkit 也能跑 GPU：在第六步 `uv sync` 之后跑 `uv run python -c "import jax; print(jax.devices())"`，输出 `[CudaDevice(id=0)]` 即说明 wheel 自带的 CUDA runtime 工作正常。

## 第五步：把仓库放到 WSL 内

**强烈建议把仓库放在 WSL 自己的文件系统里**，比如 `~/code/PowerZooJax`：

```bash
mkdir -p ~/code
cd ~/code
git clone https://github.com/powerzoojax/PowerZooJax.git
cd PowerZooJax
```

不要把仓库放在 `/mnt/c/...`、`/mnt/e/...` 这种 Windows 文件系统挂载点，原因有三：

1. 跨文件系统 IO 慢 10–100 倍，`uv sync`、`pytest`、checkpoint 写入都会被严重拖慢
2. 写 `.egg-info`、`build/`、`.uv-cache` 时容易撞 Windows 权限模型，报 `Permission denied`
3. `inotify` 文件 watcher 在 9P 协议下不可靠，`mkdocs serve` 的 livereload、PyTorch DataLoader 等都会失效

如果代码已经在 Windows 端而你不想搬：用 symlink + `TMPDIR` 折中：

```bash
ln -s /mnt/<drive>/<path-to>/PowerZooJax ~/PowerZooJax
echo 'export TMPDIR=/tmp' >> ~/.bashrc
source ~/.bashrc
cd ~/PowerZooJax
```

`TMPDIR=/tmp` 把 build 过程中 setuptools 写中间文件的目录从仓库本地（在 Windows 文件系统上）改到 WSL 自己的 tmpfs，绕开 `.egg-info` 写权限问题。但 livereload 慢的问题仍然存在；如果你长期用，建议还是搬到 `~/`。

## 第六步：用 uv 同步依赖

```bash
cd ~/code/PowerZooJax
uv sync --extra cuda12 --extra rl
```

几个细节：

- `--extra cuda12` 会装 `jax[cuda12]>=0.4.30; sys_platform == 'linux'`，sys\_platform marker 让它只在 Linux（也就是 WSL Ubuntu）上生效，原生 Windows 不会触发。
- `--extra rl` 装 rejax + distrax（单 agent PPO/SAC）；要做多 agent 用 `--extra marl` 装 jaxmarl。两者在 `pyproject.toml` 的 `[tool.uv].conflicts` 里互斥，不能同时加。
- 还可以叠加 `--extra benchmarks`（SB3 / sbx-rl）、`--extra viz`（networkx）、`--extra docs`（mkdocs 全套）、`--extra dev`（pytest）。
- 第一次 sync 要从 PyPI 拉 ~1.5 GB 的 NVIDIA wheel，耐心等。

验证 GPU 可见：

```bash
uv run python -c "import jax; print(jax.devices())"
```

期望输出：

```
[CudaDevice(id=0)]
```

如果输出是 `[CpuDevice(id=0)]`，回头检查：当前是不是在 WSL Linux 里、`--extra cuda12` 有没有传、`uv pip list | grep cuda` 里有没有 nvidia-\* wheel。

## 第七步：在 Windows 端装 VS Code 与 WSL 扩展

VS Code 装在 **Windows 端**，不是 WSL 内（WSL 内的 VS Code Server 由扩展自动装好）：

1. 从 [code.visualstudio.com](https://code.visualstudio.com/) 装 VS Code
2. 在扩展面板装：
   - `Remote - WSL`（`ms-vscode-remote.remote-wsl`）
   - `WSL`（新名字，同一个扩展）
   - `Python`（`ms-python.python`）
   - `Pylance`（`ms-python.vscode-pylance`）
   - `Ruff`（`charliermarsh.ruff`，可选）

这些扩展在 Windows 端装一次就够了；连进 WSL 时 VS Code 会提示再点一次 "Install in WSL: Ubuntu"，每个扩展只需点一次。

## 第八步：在 VS Code 里打开 WSL 中的项目

三种打开方式，挑一种：

**A. 从 Ubuntu 终端打开**（最快）

```bash
cd ~/code/PowerZooJax
code .
```

第一次运行会自动在 WSL 内装 VS Code Server（一次性，~100 MB），之后秒开。

**B. 命令面板**

VS Code 里按 `Ctrl+Shift+P` → 输入 `WSL: Open Folder in WSL...` → 选 `Ubuntu` → 在文件选择器里进 `~/code/PowerZooJax`。

**C. 左下角绿色角标**

点 VS Code 窗口左下角的绿色 `><` 图标 → `Connect to WSL using Distro...` → `Ubuntu` → `Open Folder` → `~/code/PowerZooJax`。

**验证连接成功**：左下角应显示 `WSL: Ubuntu`。从 Windows 资源管理器也能直接访问 WSL 文件，地址栏输 `\\wsl$\Ubuntu\home\<user>\code\PowerZooJax`。

打开后，给每个之前在 Windows 端装的扩展点一次 "Install in WSL: Ubuntu"，让它们在 WSL 端也就位。

## 第九步：把 VS Code 连到 uv 的 .venv

`uv sync` 会在仓库根目录建一个 `.venv/` 文件夹（位置：`~/code/PowerZooJax/.venv/`，完全在 WSL 文件系统内）。

让 VS Code 用它作为 Python 解释器：

1. `Ctrl+Shift+P` → `Python: Select Interpreter`
2. 选 `.venv/bin/python`，路径形如 `${workspaceFolder}/.venv/bin/python`

为了让这个选择固化到工作区，建议在仓库根建 `.vscode/settings.json`：

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "files.eol": "\n"
}
```

各项作用：

- `python.defaultInterpreterPath`：默认 Python 指向 `.venv`，跨机器 / 跨 clone 都生效
- `python.terminal.activateEnvironment`：新开 VS Code 终端时自动 `source .venv/bin/activate`
- `files.eol`：避免在 WSL Linux 与 Windows 之间切换造成 CRLF/LF 混乱

验证：在 VS Code 里开一个新终端，跑：

```bash
which python
# 期望: ~/code/PowerZooJax/.venv/bin/python
python -c "import jax; print(jax.devices())"
# 期望: [CudaDevice(id=0)]
```

## 第十步：运行项目 smoke test

公开提交包里用示例脚本和 benchmark 入口做验证：

```bash
uv run python examples/jax_00_verify_device.py
uv run python examples/jax_01_create_case.py
uv run python examples/jax_02_grid_env.py
```

直接检查 GPU：

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

期望 `Backend: gpu`、`Devices: [CudaDevice(id=0)]`。

第一次调用 PowerZooJax env 时可能触发 JAX JIT 编译，30–60 秒属正常，之后会很快。

## 第十一步：进阶配置

**WSL 资源上限**——在 Windows 端建 `%UserProfile%\.wslconfig`（注意是用户目录，不是项目目录）：

```ini
[wsl2]
memory=24GB
processors=8
swap=8GB
```

存盘后在 PowerShell 跑 `wsl --shutdown` 让新配置生效。

**XLA 显存预分配**——PowerZooJax 在 import 时把 `XLA_PYTHON_CLIENT_PREALLOCATE` 设成 `false`，避免 JAX 一上来就吃光显存。如果显式覆盖：

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=true   # 一次性占满，调度更稳
```

**Windows 浏览器访问 WSL 内的 mkdocs serve**——直接：

```bash
uv run mkdocs serve
```

然后在 Windows 浏览器打开 `http://localhost:8000`。WSL2 自动转发 `localhost`，不用手动配端口转发。如果 livereload 卡住，`Ctrl+C` 重启 mkdocs serve。

**Git SSH key**——放在 Ubuntu 端 `~/.ssh/`，不要复用 Windows 端的。在 Ubuntu 内：

```bash
ssh-keygen -t ed25519 -C "anonymous-key-label"
cat ~/.ssh/id_ed25519.pub  # 复制到 GitHub
```

**WSL 维护命令**（PowerShell）：

- `wsl --update`：升级 WSL 内核
- `wsl --shutdown`：关掉所有 WSL 实例（释放 RAM）
- `wsl --status`：查版本与默认发行版

## 故障排查

| 现象 | 排查 |
| --- | --- |
| `uv sync` 报 `Permission denied` 写 `.egg-info` | 仓库不要放 `/mnt/...`；如必须，`export TMPDIR=/tmp` |
| `nvidia-smi: command not found` | Windows 端装 / 升级 NVIDIA driver；`wsl --update` 后 `wsl --shutdown` 重启 |
| `nvidia-smi` 在 WSL 里能看到，但 `jax.devices()` 仅 `CpuDevice` | `--extra cuda12` 没传；或当前 shell 不在 `.venv` 里；`uv pip list \| grep cuda` 看 wheel 是否装了 |
| JAX 报 `Could not load library libcudnn.so` | `--extra cuda12` 没装齐：`uv sync --extra cuda12 --reinstall` |
| 报 cuDNN 版本不匹配 | 之前 `apt install nvidia-cuda-toolkit` 引入了系统 CUDA：`sudo apt remove nvidia-cuda-toolkit` 后重新 `uv sync --extra cuda12` |
| VS Code 里 import 失败、终端正常 | 重新执行 `Python: Select Interpreter`，选择 `.venv/bin/python` |
| WSL 占内存不释放 | PowerShell `wsl --shutdown`；用 `.wslconfig` 设上限 |
| `mkdocs serve` livereload 卡住 / 不重载 | 仓库放在 `/mnt/...` 是常见原因，移到 `~/`；或 `Ctrl+C` 重启 |
| `mkdocs serve` 浏览器打不开 | 用 `localhost:8000`，不要用 `127.0.0.1`；或 `wsl --shutdown` 重启 |
| 中文文件名 / 路径乱码 | `locale` 看 `LANG`；`sudo dpkg-reconfigure locales` 选 `zh_CN.UTF-8` 或 `en_US.UTF-8` |
| `jax.devices()` 第一次启动很慢 | 正常，CUDA 上下文初始化；后续会快 |
| `code .` 提示 `code: command not found` | Ubuntu shell 里需要先打开过一次 VS Code WSL 连接，让它把 `code` 链到 PATH |

## 下一步

- [Getting Started](../getting-started.md) —— 跑第一次 `reset` / `step` 与 scan rollout
- [Examples](../examples/index.md) —— 5 个完整 example 脚本
- [Benchmarks](../benchmarks/overview.md) —— 5 个论文任务怎么跑

要继续深入 env 层，请接着看 [Getting Started](../getting-started.md) 和示例脚本。
