# pyLDT
Python code to generate matter PDF predictions in Large Deviation Theory for LCDM and alternative cosmologies

## Installation and testing

(1) If not yet available on your machine, install julia (all platforms: download it from julialang.org; for macOS only you can alternatively 

    brew install --cask julia 
    
   with Homebrew)

(2) make sure your system has a recent pip installation by running 
    
    python -m pip install --upgrade pip

(3) for a clean install of pyLDT create a virtual environment first. I will use virtualenvwrapper, but conda or any other environment manager will do. For more details on how to install and configure virtualenvwrapper visit https://virtualenvwrapper.readthedocs.io/en/latest/index.html

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

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple pyLDT-cosmo 
    
   hopefully at this stage all remaining Python dependencies will be automatically installed too

(10) To check everything is working as expected install pytest by issuing the command

    pip install pytest 
   
   and run 
   
    pytest --pyargs pyLDT_cosmo 
   
   A test routine starts cruching the numbers (it should take about 90 sec.) and if pyLDT is correctly installed it should give 1 passed tests

## Jupyter notebook

Go to https://github.com/mcataneo/pyLDT-cosmo/tree/main and download the example jupyter notebook showing how to use pyLDT. Move the notebook into the pyLDTenv directory. To fully exploit the notebook functionalities you'll need to 'pip install matplotlib' first.

That's all! Have fun!
