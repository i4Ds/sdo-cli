from setuptools import setup, find_packages, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sdo-cli",
    version="0.0.5",
    author="Marius Giger",
    description="An ML practitioners utility for working with SDO data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where='src'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=["click>=8.1.2",
                      "python-dotenv>=0.20.0",
                      "pandas>=0.18.1"
                      "pytorch-lightning>=1.6.18.1.2",
                      "sunpy>=3.1.6",
                      "torch>=1.11.0",
                      "tqdm>=4.64.0",
                      "torchvision>=0.12.0",
                      "wandb>=0.12.15"],
    entry_points="""
        [console_scripts]
        sdo-cli=sdo.cli:cli
    """,
    url="https://github.com/i4DS/sdo-cli",
    package_dir={'': 'src'}
)
