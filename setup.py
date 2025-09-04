# author:    Kunal Kumar Kushwaha
# website:   http://www.essentia.dev

import setuptools

# Long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fp:
    install_requires = [
        line.strip() for line in fp if line.strip() and not line.startswith("#")
    ]

setuptools.setup(
    name="openbharatocr",
    version="0.5.0",
    description="OpenBharatOCR is an open-source Python library for OCR of Indian government documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/essentiasoftserv/openbharatocr",
    author="Essentia Softserv",
    author_email="kunal@essentia.dev",
    license="Apache-2.0",
    python_requires=">=3.6",
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    include_package_data=False,  # Don't automatically include LICENSE
    package_data={
        "openbharatocr": ["*.py"],
    },
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords="ocr indian documents aadhaar pan passport driving-license voter-id paddleocr easyocr tesseract",
    project_urls={
        "Bug Reports": "https://github.com/essentiasoftserv/openbharatocr/issues",
        "Source": "https://github.com/essentiasoftserv/openbharatocr",
        "Documentation": "https://github.com/essentiasoftserv/openbharatocr#readme",
    },
)
