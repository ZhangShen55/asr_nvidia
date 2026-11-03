"""Build the protected application modules as in-place Cython extensions."""

from __future__ import annotations

import glob
import os
from pathlib import Path

from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension


PROTECTED_ROOTS = ("core", "api", "utils", "entity")


def list_py(patterns, excludes=("__init__.py",)):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(
        path for path in files
        if path.endswith(".py") and os.path.basename(path) not in excludes
    )


def to_module_name(path: str) -> str:
    return os.path.splitext(path)[0].replace(os.sep, ".")


# api/**/*.py deliberately covers both api modules and nested route modules.
modules = list_py([f"{root}/**/*.py" for root in PROTECTED_ROOTS])
if not modules:
    raise RuntimeError("未发现可编译的 Cython 业务模块")

extensions = [
    Extension(
        name=to_module_name(module),
        sources=[module],
        extra_compile_args=["-std=gnu99", "-O3", "-fPIC"],
    )
    for module in modules
]

ext_modules = cythonize(
    extensions,
    nthreads=int(os.environ.get("CYTHON_NTHREADS", "4")),
    compiler_directives={
        "language_level": "3",
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "cdivision": True,
    },
)

setup(name="seacraftasr_core", ext_modules=ext_modules)
