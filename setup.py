"""
SAMS_Dunbrack
This is a toolkit to estimate free energy differences between different Dunbrack clusters of kinase conformations using self-adjusted mixture sampling (SAMS)..
"""
import sys
from setuptools import setup
import versioneer

short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:]),


setup(
    # Self-descriptive entries which should always be present
    name='sams_dunbrack',
    author='Chodera Lab // MSKCC',
    author_email='jiaye.guo@choderalab.org',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',

    # Which Python importable modules should be included when your package is installed
    packages=['sams_dunbrack', "sams_dunbrack.tests"],

    # Optional include package data to ship with your package
    # Comment out this line to prevent the files from being packaged with your software
    # Extend/modify the list to include/exclude other items as need be
    package_data={'sams_dunbrack': ["data/*.dat"]
                  },

    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,

    # Add the kinomodel as an entry point
    entry_points={
        'console_scripts': [
            'sams_dunbrack = sams_dunbrack.sams_dunbrack:main',
        ],
    }
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # author_email='me@place.org',      # Author email
    # url='http://www.my_package.com',  # Website
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    # python_requires=">=3.5",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)
