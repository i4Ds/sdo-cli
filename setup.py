from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

pkgs = find_packages(where='src')
setup(
    name="sdo-cli",
    version="0.0.19",
    author="Marius Giger",
    author_email="marius.giger@fhnw.ch",
    description="An ML practitioner's utility for working with SDO data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=pkgs,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=["beautifulsoup4>=4.11.1"
                      "click>=8.1.2",
                      "dask>=2022.5.0",
                      "drms>=0.6.2",
                      "h5netcdf>=1.0.0",
                      "matplotlib>=3.5.1",
                      "munch>=2.5.0",
                      "opencv-python>=4.5.5.64",
                      "python-dotenv>=0.20.0",
                      "pandas>=1.4.2",
                      "pytorch-lightning>=1.6.1",
                      "scikit-learn>=1.0.2",
                      "Shapely>=1.7.1",
                      "SQLAlchemy>=1.4.17",
                      "sunpy>=3.1.6",
                      "torch>=1.11.0",
                      "tqdm>=4.64.0",
                      "torchmetrics>=0.8.2",
                      "torchvision>=0.12.0",
                      "wandb>=0.12.15",
                      "zeep>=4.1.0",
                      "zarr>=2.11.3"],
    entry_points="""
        [console_scripts]
        sdo-cli=sdo.cli:cli
    """,
    url="https://github.com/i4DS/sdo-cli",
    package_dir={'': 'src'}
)
