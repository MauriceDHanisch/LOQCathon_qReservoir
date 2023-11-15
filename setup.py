# Maurice D. Hanisch mhanisc@ethz.ch
# 15.11.2023

from setuptools import setup, find_packages

setup(
    name='lo_reservoir',
    version='0.1',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)
