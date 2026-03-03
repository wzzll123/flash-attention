import os
import subprocess
import sysconfig
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import torch
import torch_npu

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class BishengBuildExt(build_ext):
    def build_extension(self, ext):
        ascend_home = os.getenv("ASCEND_TOOLKIT_HOME", os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend"))
        if not os.path.exists(ascend_home):
            raise RuntimeError(f"ASCEND_TOOLKIT_HOME={ascend_home}")

        python_include = sysconfig.get_path("include")
        torch_cmake_path = torch.utils.cmake_prefix_path
        torch_package_path = os.path.dirname(torch.__file__)
        torch_include = os.path.join(torch_cmake_path, "Torch/include")
        torch_lib = os.path.join(torch_cmake_path, "Torch/lib")

        torch_npu_path = os.path.dirname(torch_npu.__file__)
        torch_npu_include = os.path.join(torch_npu_path, "include")
        torch_npu_lib = os.path.join(torch_npu_path, "lib")

        ext_fullpath = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(ext_fullpath), exist_ok=True)

        compile_cmd = [
            "bisheng",
            "-x", "asc",
            "--npu-arch=dav-2201",
            "-shared",
            "-fPIC",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
            f"-I{ascend_home}/compiler/tikcpp/include",
            f"-I{ascend_home}/aarch64-linux/tikcpp/include",
            f"-I{python_include}",
            f"-I{torch_npu_include}",
            f"-I{torch_include}",
            f"-I{ascend_home}/include",
            f"-I{ascend_home}/runtime/include",
            f"-I{ascend_home}/include/experiment/runtime",
            f"-I{ascend_home}/include/experiment/msprof",
            f"-I{torch_package_path}/include",
            f"-I{torch_package_path}/include/torch/csrc/api/include",
            f"-I{BASE_DIR}/../csrc/catlass/include",
            f"-L{ascend_home}/compiler/lib64",
            f"-L{ascend_home}/aarch64-linux/lib64",
            f"-L{torch_lib}",
            f"-L{torch_npu_lib}",
            f"-L{torch_package_path}/lib",
            f"-L{ascend_home}/lib64",
            "-lascendcl",
            "-ltorch_npu",
            "-ltiling_api",
            "-lplatform",
            *ext.sources,
            "-o", ext_fullpath,
        ]

        print(" ".join(compile_cmd))
        subprocess.run(compile_cmd, check=True)


setup(
    name="flash_attn_npu_standalone",
    version="0.0.1",
    description="Standalone FlashAttention NPU build",
    packages=["flash_attn_npu"],
    ext_modules=[
        Extension(
            name="flash_attn_2_cuda",
            sources=[str(Path("csrc/flash_attn_npu/flash_api.cpp"))],
            language="c++",
        )
    ],
    cmdclass={"build_ext": BishengBuildExt},
)
