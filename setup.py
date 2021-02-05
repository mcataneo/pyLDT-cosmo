import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyLDT-cosmo",
    version="0.3.7",
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
