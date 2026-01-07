"""Tests for OctupleTokenizer."""

import pytest


class TestOctupleTokenizer:
    """Test cases for tokenizer."""
    
    def test_tokenizer_import(self):
        """Test that tokenizer can be imported."""
        from mamba_music.data import OctupleTokenizer
        assert OctupleTokenizer is not None
