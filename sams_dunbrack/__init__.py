"""
SAMS_Dunbrack
This is a toolkit to estimate free energy differences between different Dunbrack clusters of kinase conformations using self-adjusted mixture sampling (SAMS)..
"""

# Add imports here
from .sams_dunbrack import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
