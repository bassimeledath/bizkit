import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bizkit",
    version="0.0.4",
    author="Bassim Eledath, Lynn He, Christine Zhu, Amanda Ma",
    author_email="bassimfaizal@gmail.com",
    description="A package that streamlines business data analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/bizkit",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas",
	    "numpy",
        "lifelines",
        "bokeh",
	    "sklearn",
	    "mlxtend",
	    "d3fdgraph"

    ]

)
