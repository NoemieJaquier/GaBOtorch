from setuptools import setup, find_packages

# get description from readme file
with open('README.md', 'r') as f:
    long_description = f.read()

# setup
setup(
    name='BoManifolds_torch',
    version='',
    description='',
    long_description = long_description,
    long_description_content_type="text/markdown",
    author='Noémie Jaquier, Leonel Rozo',
    author_email='noemie.jaquier@de.bosch.com, leonel.rozo@de.bosch.com',
    maintainer=' ',
    maintainer_email='',
    license=' ',
    url=' ',
    platforms=['Linux Ubuntu'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
