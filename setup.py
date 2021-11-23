import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="nephosem",
    version="0.1.1",
    author="QLVL",
    description = "Python package for creating type and token level distributional models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPL-3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    py_modules=["nephosem"],
    package_dir={'':'nephosem/'},
    install_requires=[]
)