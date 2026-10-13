"""Build the application as an executable folder; JSON stays outside the archive."""

from pathlib import Path

root = Path(SPECPATH)
a = Analysis(
    [str(root / "scripts/frozen_entry.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "nnunetv2", "tkinter", "IPython", "pytest", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [("u", None, "OPTION"), ("X utf8", None, "OPTION")],
    exclude_binaries=True,
    name="AutoPrintFilm",
    console=True,
    strip=False,
    upx=False,
    contents_directory="_internal",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="AutoPrintFilm")
