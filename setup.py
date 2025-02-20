from setuptools import setup, find_packages

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