from setuptools import setup, find_packages
from pathlib import Path

package_dir = Path(Path(__file__).parent).resolve()

long_description = ""
with Path(package_dir, "README.md").open("r") as fh:
    long_description = fh.read()
assert long_description

VERSION = ""
with Path(package_dir, "stanza_batch", "version.py").open("r") as fh:
    temp_version = {}
    exec(fh.read(), None, temp_version)
    VERSION = temp_version["VERSION"]
assert VERSION

setup(
    name="stanza_batch",
    version=VERSION,
    description="A batching utility for Stanza",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apmoore1/stanza-batch",
    author="Andrew Moore",
    author_email="andrew.p.moore94@gmail.com",
    license="Apache License 2.0",
    install_requires=[
        "stanza>=1.1.1,<=1.2.0",
    ],
    python_requires=">=3.6.1",
    packages=find_packages(include=["stanza_batch"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    # Came from the Stanza setup.py
    keywords="natural-language-processing nlp natural-language-understanding stanford-nlp deep-learning",
)
