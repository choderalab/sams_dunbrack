"""
Unit and regression test for the sams_dunbrack package.
"""

# Import package, test suite, and other packages as needed
import sams_dunbrack
import pytest
import sys

def test_sams_dunbrack_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "sams_dunbrack" in sys.modules
