import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch

cxx_flags = []
ext_libs = []

authors = [
        'Jiaao He', 
        'Jiezhong Qiu', 
        'Aohan Zeng', 
        'Tiago Antunes', 
        'Jinjun Peng', 
        'Qin Li',
        'Mingshu Zhai',
        'Yuntao Nie'
]

is_rocm_pytorch = False
if torch.__version__ >= '1.5':
    from torch.utils.cpp_extension import ROCM_HOME
    is_rocm_pytorch = True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False

if os.environ.get('USE_NCCL', '1') == '1':
    cxx_flags.append('-DFMOE_USE_NCCL')
    cxx_flags.append('-DUSE_C10D_NCCL')
    if is_rocm_pytorch:
        ext_libs.append('rccl')
    else:
        ext_libs.append('nccl')

if os.environ.get('MOE_DEBUG', '0') == '1':
    cxx_flags.append('-DMOE_DEBUG')

if is_rocm_pytorch:
    define_macros=[('FMOE_USE_HIP', None)]
else:
    define_macros=[]


if __name__ == '__main__':
    setuptools.setup(
        name='magi',
        version='1.0.0',
        description='An efficient MoE expert parallel training system base on FastMoE and Megatron',
        author=', '.join(authors),
        author_email='hja20@mails.tsinghua.edu.cn,nieyuntao@hust.edu.cn',
        license='Apache-2',
        url='https://github.com/laekov/fastmoe,https://github.com/GITD245/Magi',
        packages=['fmoe', 'fmoe.megatron', 'fmoe.gates', 'fmoe.magi_schedule', 'magi', 'magi.megatron'],
        ext_modules=[
            CUDAExtension(
                name='fmoe_cuda', 
                sources=[
                    'cuda/stream_manager.cpp',
                    'cuda/local_exchange.cu',
                    'cuda/balancing.cu',
                    'cuda/global_exchange.cpp',
                    'cuda/parallel_linear.cu',
                    'cuda/fmoe_cuda.cpp',
                    'cuda/magi/smart_schedule.cpp',
                    ],
                define_macros=define_macros,
                extra_compile_args={
                    'cxx': cxx_flags,
                    'nvcc': cxx_flags
                    },
                libraries=ext_libs
                )
            ],
        cmdclass={
            'build_ext': BuildExtension
        })
