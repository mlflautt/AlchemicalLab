from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cellularautomatalab",
    version="0.1.0",
    author="Mitchell Flautt",
    description="A comprehensive cellular automata research and experimentation framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mitchellflautt/CellularAutomataLab",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Life",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pillow>=8.3.0",
        "pygame>=2.0.1",
        "numba>=0.54.0",
        "networkx>=2.6",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.4",
            "black>=21.6b0",
            "flake8>=3.9.2",
            "mypy>=0.910",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipython>=7.25.0",
        ],
    },
)