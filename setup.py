import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="data_tools",
    version="0.0.1",
    description="A Dataset Utilities Library for Pandas",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/hsteinshiromoto/cartorio",
    author="Humberto STEIN SHIROMOTO",
    author_email="h.stein.shiromoto@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=["data_tools"],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "data_tools=data_tools.__main__:main",
        ]
    },
)
