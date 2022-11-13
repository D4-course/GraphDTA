"""
Simple testing file that ensures the drug inputted is in proper
SMILES format and the protein sequence is proper
"""
import pytest
from create_data import verify_smile, verify_protein

def test_format_input_1():
    assert verify_smile("OCCn1cc(-c2ccc3c(c2)CCC3=NO)c(-c2ccncc2)n1") == True

def test_format_input_2():
    assert verify_smile("This is not a molecule") == False

def test_format_input_1():
    assert verify_protein("ACDDGH") == True

def test_format_input_2():
    assert verify_smile("XYZ") == False