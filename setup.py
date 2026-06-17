from setuptools import setup, find_packages

setup(
    name='dexmani_real',
    packages=find_packages(),
    package_data={
        'dexmani_real': ['py.typed'],
    },
    python_requires='>=3.10',
)