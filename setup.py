from setuptools import setup, find_packages

setup(
    name='ReMoNet',
    version='0.1.0',
    description='ReMoNet: Regridding Tools for xarray',
    author='bbakernoaa',
    author_email='your-email@example.com',
    url='https://github.com/bbakernoaa/ReMONET',
    packages=find_packages(),
    install_requires=[
        'xarray',
        'numpy',
        'scipy',
        'dask'
    ],
    extras_require={
        'cf': ['cf_xarray']
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
