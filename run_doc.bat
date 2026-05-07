@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Local MkDocs runner for PowerZooJax.
rem   run_doc.bat                Start local dev server on 127.0.0.1:8000
rem   run_doc.bat serve          Same as above
rem   run_doc.bat build          One-shot strict build to site/
rem   run_doc.bat --dirty        Serve with --dirty (faster reloads, but breaks
rem                              sub-pages under mkdocs-static-i18n; avoid)
rem   run_doc.bat --stable       No-op alias for backwards compat (default now)
rem   run_doc.bat -a 8001        Shorthand: same as -a 127.0.0.1:8001
rem   run_doc.bat serve --dev-addr 127.0.0.1:8001
rem
rem We intentionally use a docs-only uv environment instead of
rem `uv run --extra docs ...`, because the project dependencies include
rem CUDA-specific JAX packages that do not resolve on some local setups.

cd /d "%~dp0"

set "command=serve"
set "dirty=0"
set /a extra_count=0

:parse_args
if "%~1"=="" goto normalize_args
if /i "%~1"=="serve" (
  set "command=serve"
  shift
  goto parse_args
)
if /i "%~1"=="build" (
  set "command=build"
  shift
  goto parse_args
)
if "%~1"=="--dirty" (
  set "dirty=1"
  shift
  goto parse_args
)
if "%~1"=="--stable" (
  rem Default behavior; kept for backwards compat.
  shift
  goto parse_args
)
set /a extra_count+=1
set "extra[!extra_count!]=%~1"
shift
goto parse_args

:normalize_args
set "extra_args="
set /a i=1

:normalize_loop
if !i! gtr !extra_count! goto setup_env
set "x=!extra[%i%]!"

if /i "!x!"=="-a" goto normalize_dev_addr
if /i "!x!"=="--dev-addr" goto normalize_dev_addr

set "extra_args=!extra_args! "!x!""
set /a i+=1
goto normalize_loop

:normalize_dev_addr
set "extra_args=!extra_args! "!x!""
set /a i+=1
if !i! gtr !extra_count! goto setup_env
set "val=!extra[%i%]!"
echo(!val!| findstr /r /x "[0-9][0-9]*" >nul
if not errorlevel 1 (
  set "val=127.0.0.1:!val!"
)
set "extra_args=!extra_args! "!val!""
set /a i+=1
goto normalize_loop

:setup_env
if not exist "docs\.jupyter_site_build" mkdir "docs\.jupyter_site_build"
if not exist "docs\zh" mkdir "docs\zh"

set "JUPYTER_CONFIG_DIR=%CD%\docs\.jupyter_site_build"
set "JUPYTER_DATA_DIR=%CD%\docs\.jupyter_site_build"
if defined PYTHONPATH (
  set "PYTHONPATH=%CD%;%PYTHONPATH%"
) else (
  set "PYTHONPATH=%CD%"
)

set "uv_mkdocs=uv run --no-project --with mkdocs --with mkdocs-material --with mkdocs-static-i18n --with mkdocstrings-python --with pymdown-extensions mkdocs"

if /i "%command%"=="build" (
  call %uv_mkdocs% build --strict%extra_args%
  exit /b %errorlevel%
)

set "mkdocs_args=serve --livereload"
if "%dirty%"=="1" (
  set "mkdocs_args=%mkdocs_args% --dirty"
)

call %uv_mkdocs% %mkdocs_args%%extra_args%
exit /b %errorlevel%
