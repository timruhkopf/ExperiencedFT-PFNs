from setuptools import setup, find_packages
import pathlib

def parse_requirements(file_path: str):
    """Load requirements from requirements.txt"""
    requirements = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line:
                continue
            # Handle -r includes (optional)
            if line.startswith('-r'):
                continue  # For simplicity, we'll ignore nested requirements
            # Remove version specifiers for setup.py
            requirements.append(line.split(';')[0].split('#')[0].strip())
    return requirements

# Read long description from README
here = pathlib.Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8")
requirements = parse_requirements(here / 'requirements.txt')

setup(
    name="ExperiencedFTPFN",
    version="0.4.1",
    description="In-Context Distillation for warm starting FTPFN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/yourusername/ml-logger",
    author="Tim Ruhkopf",
    author_email="timruhkopf@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="machine-learning logging visualization mlops",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11.0",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        # "docs": [
        #     "sphinx>=4.0",
        #     "sphinx-rtd-theme>=0.5.0",
        # ],
    },
    # project_urls={
    #     "Bug Reports": "https://github.com/yourusername/ml-logger/issues",
    #     "Source": "https://github.com/yourusername/ml-logger",
    # },
)
