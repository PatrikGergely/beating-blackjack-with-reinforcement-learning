from setuptools import setup
from Cython.Build import cythonize
import numpy as np


setup(
    ext_modules=cythonize(
        ["bbwrl/bot/reward_distribution.pyx",
         "bbwrl/bot/strategists/optimal_strategist.pyx",
         "bbwrl/bot/bettors/kelly_bettor.pyx"],
        #compiler_directives={'linetrace': True},
        annotate=True),
        include_dirs=[np.get_include()]
)
