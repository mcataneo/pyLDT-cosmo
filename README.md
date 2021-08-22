# pyLDT-cosmo
A Python package to generate matter PDF predictions in Large Deviation Theory for ΛCDM and alternative cosmologies

<img src="docs/pdfs_z.png" width="400"> <img src="docs/pdfs_R.png" width="400">

## Installation

(1) If not yet available on your machine, install julia (all platforms: download it from julialang.org; for macOS only you can alternatively 

    brew install --cask julia 
    
   with Homebrew)

(2) make sure your system has a recent pip installation by running 
    
    python -m pip install --upgrade pip

(3) for a clean install of pyLDT create a virtual environment first. I will use virtualenvwrapper, but conda or any other environment manager will do ([see installation note below for conda users](#conda_footnote)). For more details on how to install and configure virtualenvwrapper visit https://virtualenvwrapper.readthedocs.io/en/latest/index.html

(4) Once virtualenvwrapper is setup, create simultaneously a project and an environment (e.g., pyLDTenv) typing in terminal

    mkproject pyLDTenv 
   
   If the envornment is not yet activated, type 
   
    workon pyLDTenv 
   
   This should take you directly into the pyLDTenv directory associated with the pyLDTenv project. 

(5) Install PyJulia by running 

    python3 -m pip install julia

(6) To install the Julia packages required by PyJulia launch a Python REPL and run the following code 

    >>> import julia 
    >>> julia.install() 

(7) Install diffeqpy by running 

    pip install diffeqpy

(8) To install Julia packages required for diffeqpy, open up the Python interpreter and run

    >>> import diffeqpy
    >>> diffeqpy.install()

(9) Now run 

    pip install pyLDT-cosmo 
    
   hopefully at this stage all remaining Python dependencies will be automatically installed too

(10) To check everything is working as expected install pytest by issuing the command

    pip install pytest 
   
   and run 
   
    pytest --pyargs pyLDT_cosmo 
   
   A test routine starts cruching the numbers (it should take about 80 sec.) and if pyLDT-cosmo is correctly installed it should give 1 passed tests

<a name="conda_footnote"></a>
### Note for Conda users

First make sure to install pip in your conda environment with

    conda install pip

Replace all subsequent 'pip install' commands with 'python -m pip install' so that packages are installed using the virtual enviroment's pip.

Owing to an [incompatibility between PyJulia and Conda](https://pyjulia.readthedocs.io/en/latest/troubleshooting.html), PyJulia cannot be properly initialised out-of-the-box. As a workaround pyLDT-cosmo will automatically disable the precompilation cache mechanism in Julia, which inevitably slows down loading and usage of Julia packages. As a result, loading pyLDT-cosmo can take up to 3x longer in a conda envirnoment and PDF calculations can easily double their execution time. 

## Models

Currently available cosmological models include:

* ΛCDM
* Hu-Sawicki f(R) gravity with ΛCDM background ([0705.1158](https://arxiv.org/abs/0705.1158)) 
* nDGP gravity with ΛCDM background ([0910.0235](https://arxiv.org/abs/0910.0235))
* w0waCDM ([0009008](https://arxiv.org/abs/gr-qc/0009008), [0208512](https://arxiv.org/abs/astro-ph/0208512), [0808.3125](https://arxiv.org/abs/0808.3125))

Einstein-de Sitter spherical evolution is assumed for all cases.

If interested in implementing other modified gravity models: 

* Follow the installation steps (1)-(8) above
* Clone this Git repo into the newly created environment
* Move into the pyLDT-cosmo directory and install pyLDT-cosmo in developer (or editable) mode with

        pip install -e .

* Add the relevant linear theory equations and methods to the following modules in the 'pyLDT_cosmo' sub-directory:

    * growth_eqns.py
    * solve_eqns.py
    * compute_pk.py
    * pyLDT.py

  For an example, track f(R) gravity (or nDGP gravity).

## Jupyter notebook

Go to https://github.com/mcataneo/pyLDT-cosmo/tree/main and download the example jupyter notebook showing how to use pyLDT-cosmo. Move the notebook into the pyLDTenv directory. To fully exploit the notebook functionalities you'll need to 'pip install matplotlib' first.

### A word on loading and computing time
pyLDT-cosmo is partly based on the Julia programming language, which uses a Just-In-Time (JIT) compiler to improve runtime performance. However, this feature comes at the expense of slow library loading when compared to pure or pre-compiled Python packages. On a modern computer pyLDT-cosmo takes about 80 seconds to load. After that computation is very efficient, taking only ~3 seconds to derive the matter PDF's simultaneuosly for all models, redshifts and smoothing radii.  

That's all! Have fun!
