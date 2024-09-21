import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pta_priors",
    version="0.0.1",
    author="Boris Goncharov",
    author_email="goncharov.boris@physics.msu.ru",
    description="Prior modelling for PTA analyses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bvgoncharov/pta_priors",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
