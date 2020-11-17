from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="stanza_batch",
    version="0.1.0",
    description="A batching utility for Stanza",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apmoore1/stanza-batch",
    author="Andrew Moore",
    author_email="andrew.p.moore94@gmail.com",
    license="Apache License 2.0",
    install_requires=[
        "stanza>=1.1.1",
    ],
    python_requires=">=3.6.1",
    packages=find_packages(include=["stanza_batch"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
    ],
    # Came from the Stanza setup.py
    keywords="natural-language-processing nlp natural-language-understanding stanford-nlp deep-learning",
)
