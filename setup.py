import setuptools
import os
import re

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(os.path.join(os.path.dirname(__file__), 'pyLDT_cosmo', '__init__.py')) as fp:
    for line in fp:
        m = re.search(r'^\s*__version__\s*=\s*([\'"])([^\'"]+)\1\s*$', line)
        if m:
            version = m.group(2)
            break
    else:
        raise RuntimeError('Unable to find own __version__ string')

setuptools.setup(
    name="pyLDT-cosmo",
    version=version,
    author="Matteo Cataneo",
    author_email="mcataneo85@gmail.com",
    description="Package for PDF calculations in Large Deviation Theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mcataneo/pyLDT-cosmo",
    packages=setuptools.find_packages(),
    package_data={'pyLDT_cosmo': ['benchmarks/*.dat']},
    install_requires=[
        'numpy>=1.19',
        'scipy>=1.5',
        'camb==1.1.2',
        'mcfit==0.0.16',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
