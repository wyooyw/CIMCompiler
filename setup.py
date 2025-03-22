from setuptools import setup, find_packages
import subprocess
import os

def run_scripts():
    # Run the llvm_build.sh script in its directory
    subprocess.run(['bash', 'llvm_build.sh'], cwd='thirdparty/llvm-project', check=True)

    # Run the build.sh script in the current directory
    subprocess.run(['bash', './build.sh'], check=True)

# Call the function to run the scripts
run_scripts()

setup(
    name="cim_compiler",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'cim-compiler=cim_compiler.cli.main:main',
            'cmcp=cim_compiler.cli.main:main',
        ],
    },
    install_requires=[
        # 在这里列出你的依赖包
    ],
)