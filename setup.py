from setuptools import find_packages, setup

setup(
    name="dexmani_real",
    packages=find_packages(),
    package_data={
        "dexmani_real": ["py.typed"],
    },
    python_requires=">=3.10",
)
