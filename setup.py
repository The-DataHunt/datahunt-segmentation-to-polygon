from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Converting mask to polygon'
LONG_DESCRIPTION = 'A package that makes it to convert segmentation masking data to polygon'

setup(
    name="mask_to_polygon",
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author="<Jisu Yu>",
    author_email="<jisu.yu@thedatahunt.com>",
    license='',
    packages=find_packages(),
    install_requires=[],
    keywords='algorithm'
)
