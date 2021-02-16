from setuptools import setup, find_packages

setup(
    name="sdo-cli",
    version="1.0",
    packages=["sdo", "sdo.cmd", "sdo.cmd.data", "sdo.data_loader.image_param"],
    include_package_data=True,
    install_requires=["click"],
    entry_points="""
        [console_scripts]
        sdo-cli=sdo.cli:cli
    """,
)
