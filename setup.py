# author:    Kunal Kumar Kushwaha
# website:   http://www.essentia.dev

import setuptools
import glob

# Long description
with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fp:
    install_requires = fp.read()

setuptools.setup(
    name="openbharatocr",
    version="0.3.2",
    description="openbharatocr is an opensource python library for ocr Indian government documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/essentiasoftserv/openbharatocr",
    author="essentiasoftserv",
    python_requires=">=3.6",
    install_requires=install_requires,
    author_email="kunal@essentia.dev",
    packages=setuptools.find_packages(),
    include_package_data=True,
    zip_safe=False,
    license="Apache",
)
