"""ai4data.metadata.augmentation — LLM-powered data dictionary augmentation.

Automatically generates thematic structure for microdata or administrative
data dictionary variables using semantic clustering and LLM-elicited themes.

Install
-------
.. code-block:: bash

    uv pip install ai4data[metadata]

Quick Start
-----------
.. code-block:: python

    from ai4data.metadata.augmentation import DataDictionaryAugmentor

    augmentor = DataDictionaryAugmentor()
    result = augmentor.augment("variables.csv")
    augmentor.export("augmented.json")

The pipeline: Load → Embed → Cluster → Generate Themes → Export.

See :class:`DataDictionaryAugmentor` for the full API.
"""

from . import adapters, clustering, embeddings, prompts, schemas
from .adapters import (
    ConfigurableDictionaryAdapter,
    NADACatalogAdapter,
    adapter_from_config,
)
from .augmentor import DEFAULT_MODEL, DataDictionaryAugmentor
from .schemas import (
    AugmentedDictionary,
    DictionaryVariable,
    Theme,
    ThemeAssignment,
    ThemeGenerationResult,
)

__all__ = [
    # Main class
    "DataDictionaryAugmentor",
    "DEFAULT_MODEL",
    # Adapters
    "ConfigurableDictionaryAdapter",
    "NADACatalogAdapter",
    "adapter_from_config",
    # Schemas
    "AugmentedDictionary",
    "DictionaryVariable",
    "Theme",
    "ThemeAssignment",
    "ThemeGenerationResult",
    # Submodules
    "adapters",
    "clustering",
    "embeddings",
    "prompts",
    "schemas",
]
