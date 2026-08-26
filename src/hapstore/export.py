"""Compatibility imports for the former generic export module.

Use :mod:`hapstore.export_vcf` for VCF export and :mod:`hapstore.subset` for
sample-axis store copying in new code. Top-level imports from :mod:`hapstore`
remain the supported public API.
"""

from .export_vcf import export_vcf
from .subset import subset_samples_to_new_store

__all__ = ["export_vcf", "subset_samples_to_new_store"]
