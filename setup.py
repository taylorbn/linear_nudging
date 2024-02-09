from setuptools import setup, find_packages

setup(
    name="nudge",
    version=0.1,
    description="Runs nudging experiments for linear 2x2 systems",
    author="Nathan Taylor",
    packages=find_packages(),
    install_requires=[
        "click",
    ],
    scripts=[
        "bin/nudge",
    ],
)
