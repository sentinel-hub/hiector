import os
from setuptools import setup, find_packages


def parse_requirements(file):
    with open(os.path.join(os.path.dirname(__file__), file)) as req_file:
        return [line.strip() for line in req_file if "/" not in line]


def get_version():
    for line in open(os.path.join(os.path.dirname(__file__), "hiector", "__init__.py")):
        if line.find("__version__") >= 0:
            version = line.split("=")[1].strip()
            version = version.strip('"').strip("'")
    return version


setup(
    name="hiector",
    python_requires=">=3.7",
    version=get_version(),
    description="HIErarchical deteCTOR, a package for hierarchical building detection",
    url="https://github.com/sentinel-hub/hiector",
    author="Sinergise EO research team",
    author_email="eoresearch@sinergise.com",
    license="MIT",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
    extras_require={"DEV": parse_requirements("requirements-dev.txt")},
    zip_safe=False,
    keywords="building, object detection, hierarchical",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Build Tools",
    ],
)
