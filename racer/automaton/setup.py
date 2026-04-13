from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        name="Automaton",
        sources=["automaton.pyx", "src/automaton.cpp"],
        language="c++",
        include_dirs=["."],
        extra_compile_args=["-std=c++17", "-O2"],
    )
]

setup(
    ext_modules=cythonize(ext_modules)
)