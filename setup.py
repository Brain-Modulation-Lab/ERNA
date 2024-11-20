from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    install_requires = f.read().strip().splitlines()
    
setup(
    name='ERNA_GUI', #needs to build fabric
    version='0.1.0',
    description='A library to analyze and visualize ERNA signals.',
    packages=find_packages(),  
    author= "MatteoV",
    author_email= "mvissani@mgh.harvard.edu",
    license="APACHE v2.0",
    python_requires='>=3.8',
    install_requires=install_requires,  # Use the requirements from requirements.txt
    entry_points={
        'console_scripts': [
            'ERNA_viz = ERNA_GUI.viz.GUI:launch_streamlit',
            'ERNA_api = ERNA_GUI.api.ERNA:main',
        ],
    },
)