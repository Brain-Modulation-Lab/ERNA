from setuptools import setup, find_packages

setup(
    name='ERNA_GUI', #needs to build fabric
    version='0.1.0',
    description='A library to analyze and visualize ERNA signals.',
    packages=find_packages(),  
    author= "MatteoV",
    author_email= "mvissani@mgh.harvard.edu",
    license="APACHE v2.0"
)