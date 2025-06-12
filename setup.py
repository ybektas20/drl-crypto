from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt', 'r') as file:
        return [line.strip() for line in file if line.strip() and not line.startswith('#')]


setup(
    name="drl-crypto",
    version="1.0.0",
    packages=find_packages(),
    install_requires=read_requirements(),
    author="Yusuf Emir Bektaş",
    author_email="ybektas20@ku.edu.tr",
    description="deep rl for portfolio optimization in crypto markets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)