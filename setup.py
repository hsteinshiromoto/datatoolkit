import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

docs_extras = [
    "Sphinx >= 3.0.0",  # Force RTD to use >= 3.0.0
    "docutils",
    "pylons-sphinx-themes >= 1.0.8",  # Ethical Ads
    "pylons_sphinx_latesturl",
    "repoze.sphinx.autointerface",
    "sphinxcontrib-autoprogram",
]

# This call to setup() does all the work
setup(
    name="datatoolkit",
    version="0.2.5",
    description="A collection of tools for visualization and data processing for exploratory data analysis.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/hsteinshiromoto/datatoolkit",
    author="Humberto STEIN SHIROMOTO",
    author_email="h.stein.shiromoto@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["datatoolkit"],
    include_package_data=True,
    install_requires=[
        "seaborn",
        "typeguard",
        "statsmodels",
        "bokeh",
        "networkx",
        "hyperopt",
    ],
    entry_points={
        "console_scripts": [
            "datatoolkit=datatoolkit.__main__:main",
        ]
    },
    extras_require={"docs": docs_extras},
)
