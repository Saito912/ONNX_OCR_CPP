import sys
from setuptools import setup, Extension
import pybind11

# 根据平台设置 OpenMP 标志
# GCC/Clang 使用 -fopenmp
# MSVC (Windows) 使用 /openmp
if sys.platform == 'win32':
    openmp_args = ['/openmp']
else:
    openmp_args = ['-fopenmp']

ext_modules = [
    Extension(
        'ctc_decoder',
        ['src/ctc_decode.cpp'], # 确保这里是你的新 C++ 文件名
        include_dirs=[
            pybind11.get_include(),
        ],
        language='c++',
        # 为编译器和链接器添加 OpenMP 标志
        extra_compile_args=['-std=c++14', '-O3'] + openmp_args if sys.platform != 'win32' else ['/EHsc', '/O2'] + openmp_args,
        extra_link_args=openmp_args
    ),
]

setup(
    name='ctc_decoder',
    version='0.2.0', # 建议增加版本号
    author='Your Name',
    author_email='your.email@example.com',
    description='An optimized pybind11 wrapper for a C++ CTC decoder',
    long_description='A fast, parallelized C++ implementation of CTC beam search decoding, wrapped for Python using pybind11.',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6'],
    zip_safe=False,
    python_requires='>=3.6',
)