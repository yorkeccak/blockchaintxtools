from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="blockchaintxtools",
    version="0.1.1",
    author="Harvey Yorke",
    author_email="harvey@valyu.network",
    description="A comprehensive toolkit for collecting, analyzing, filtering, and exporting blockchain transactions using Blockchair's database dumps.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yorkeccak/blockchaintxtools",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.7",
    install_requires=[
        "requests",
        "pandas",
        "tqdm",
        "pydantic",
        "python-dotenv",
    ],
)