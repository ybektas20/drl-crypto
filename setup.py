from setuptools import setup, find_packages
from pathlib import Path

def read_requirements():
    return Path("requirements.txt").read_text().splitlines()

setup(
    name="drl-crypto",
    version="1.0.0",
    packages=find_packages(exclude=["tests", "tests.*"]),  
    install_requires=read_requirements(),
    author="Yusuf Emir Bektaş",
    author_email="ybektas20@ku.edu.tr",
    description="Deep RL for portfolio optimisation in crypto markets",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    python_requires=">=3.12",
    include_package_data=True,
)
