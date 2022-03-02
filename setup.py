import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="norm",
    version="0.0.1",
    author="Danilo Numeroso",
    author_email="danilo.numeroso@phd.unipi.it",
    description="Set of utilities for managing ML experiments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danilonumeroso/norm",
    project_urls={
        "Bug Tracker": "https://github.com/danilonumeroso/norm",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "code"},
    packages=setuptools.find_packages(where="code"),
    python_requires=">=3.8",
    install_requires=[
        "matplotlib >= 3.0",
        "numpy >= 1.21",
        "randomname",
        "ray >= 1.9.2"
    ]
)
