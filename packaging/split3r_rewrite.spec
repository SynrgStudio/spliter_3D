# -*- mode: python ; coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
if ROOT.name == "packaging":
    ROOT = ROOT.parent

APP_NAME = "Split3rRewrite"
_datas, _binaries, _hiddenimports = [], [], []


def collect(package: str) -> None:
    try:
        d, b, h = collect_all(package)
    except Exception:
        d, b, h = [], [], collect_submodules(package)
    _datas.extend(d)
    _binaries.extend(b)
    _hiddenimports.extend(h)


for package in ("PyQt6", "vtk", "vtkmodules", "pyvista", "trimesh", "networkx", "lxml", "numpy"):
    collect(package)

_hiddenimports.extend(collect_submodules("split3r_rewrite"))

_a = Analysis(
    [str(ROOT / "split3r_rewrite" / "app.py")],
    pathex=[str(ROOT)],
    binaries=_binaries,
    datas=_datas,
    hiddenimports=sorted(set(_hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "tests", "tests_rewrite", "cv2", "moviepy", "speech_recognition"],
    noarchive=False,
    optimize=0,
)
_pyz = PYZ(_a.pure)
_exe = EXE(
    _pyz,
    _a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)
_coll = COLLECT(_exe, _a.binaries, _a.datas, strip=False, upx=True, name=APP_NAME)
