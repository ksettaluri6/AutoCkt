"""
# Setup Script

Derived from the setuptools sample project at
https://github.com/pypa/sampleproject/blob/main/setup.py

"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
readme = here / "readme.md"
long_description = "" if not readme.exists() else readme.read_text(encoding="utf-8")


setup(
    name="autockt",
    version="0.0.1",
    description="BWRC AMS ML Discovery AutoCkt - ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="FIXME",
    author="The Regents of the University of California",
    author_email="FIXME",
    packages=find_packages(),
    python_requires=">=3.7, <3.8",  ## FIXME: require 3.7, maybe more, after dependencies upgrades
    install_requires=[  ##
        ## FIXME: can we ease up on the version requirements?
        ## Maybe, but it's nice for intra-workspace consistency.
        "numpy==1.16.4",
        "scipy==1.1.0",
        "gym==0.10.5",  # Core ML dependency: OpenAI Gym
        "ray==0.6.3",  # Ray for RL, parallelization, training
        "tensorflow==1.13.1",  # "ray" needs these
        "protobuf==3.20.0",  # "ray" needs these
        "opencv-python==4.6.0.66",  # "ray" needs these
        "ipython==6.5.0",  # FIXME: probably dev only?
        "pyyaml==5.1.2",
        "autockt_shared",  # Local "workspace" dependency
    ],
    # extras_require={
    #     "dev": [
    #         "pytest==7.1",
    #         "coverage",
    #         "pytest-cov",
    #         "pre-commit==2.20",
    #         "black==22.6",
    #         "twine",
    #     ]
    # },
)
