#!/usr/bin/env bash
# Local MkDocs runner for PowerZooJax.
#   ./run_doc.sh                Start local dev server on 127.0.0.1:8000
#   ./run_doc.sh serve          Same as above
#   ./run_doc.sh build          One-shot strict build to site/
#   ./run_doc.sh --dirty        Serve with --dirty (faster reloads, but breaks
#                               sub-pages under mkdocs-static-i18n; avoid)
#   ./run_doc.sh --stable       No-op alias for backwards compat (default now)
#   ./run_doc.sh -a 8001        Shorthand: same as -a 127.0.0.1:8001
#   ./run_doc.sh serve --dev-addr 127.0.0.1:8001
#
# We intentionally use a docs-only uv environment instead of
# `uv run --extra docs ...`, because the project dependencies include
# CUDA-specific JAX packages that do not resolve on macOS.

set -euo pipefail

cd "$(dirname "$0")"

command="serve"
dirty=0
args=()

for a in "$@"; do
  case "$a" in
    serve|build)
      command="$a"
      ;;
    --dirty)
      dirty=1
      ;;
    --stable)
      # Default behavior; kept for backwards compat.
      ;;
    *)
      args+=("$a")
      ;;
  esac
done

# MkDocs requires dev_addr as IP:PORT; accept port-only shorthand (e.g. -a 8002).
normalize_mkdocs_extra_args() {
  local out=()
  local i=0
  while [ "$i" -lt "${#args[@]}" ]; do
    local x="${args[$i]}"
    if [[ "$x" == "-a" || "$x" == "--dev-addr" ]]; then
      out+=("$x")
      i=$((i + 1))
      if [ "$i" -lt "${#args[@]}" ]; then
        local val="${args[$i]}"
        if [[ "$val" =~ ^[0-9]+$ ]]; then
          out+=("127.0.0.1:${val}")
        else
          out+=("$val")
        fi
        i=$((i + 1))
      fi
    else
      out+=("$x")
      i=$((i + 1))
    fi
  done
  # Bash 3.2 + set -u: empty `"${array[@]}"` errors; branch instead of copying empty.
  if [ "${#out[@]}" -eq 0 ]; then
    args=()
  else
    args=("${out[@]}")
  fi
}
normalize_mkdocs_extra_args

mkdir -p docs/.jupyter_site_build
mkdir -p docs/zh

export JUPYTER_CONFIG_DIR="${PWD}/docs/.jupyter_site_build"
export JUPYTER_DATA_DIR="${PWD}/docs/.jupyter_site_build"
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

uv_mkdocs=(
  uv run
  --no-project
  --with mkdocs
  --with mkdocs-material
  --with mkdocs-static-i18n
  --with mkdocstrings-python
  --with pymdown-extensions
  mkdocs
)

if [ "$command" = "build" ]; then
  if [ "${#args[@]}" -eq 0 ]; then
    exec "${uv_mkdocs[@]}" build --strict
  else
    exec "${uv_mkdocs[@]}" build --strict "${args[@]}"
  fi
fi

mkdocs_args=(serve --livereload)
if [ "$dirty" -eq 1 ]; then
  mkdocs_args+=(--dirty)
fi

if [ "${#args[@]}" -eq 0 ]; then
  exec "${uv_mkdocs[@]}" "${mkdocs_args[@]}"
else
  exec "${uv_mkdocs[@]}" "${mkdocs_args[@]}" "${args[@]}"
fi
