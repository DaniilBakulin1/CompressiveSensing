from setuptools import setup, find_packages

setup(
    name="csmp",
    version="0.1.0",
    author="xephosbot",
    description="Library for compressive sensing matching pursuit",
    packages=find_packages(),
    install_requires=["numpy", "scipy"],  # Зависимости
)