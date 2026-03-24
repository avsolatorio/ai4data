"""Embedding utilities for data dictionary variables.

Uses ``sentence-transformers`` to encode variable labels and descriptions
into dense vectors for semantic clustering. The model is loaded lazily
on first use to avoid import-time overhead when the package is loaded
but embeddings are not needed.

Default model: ``BAAI/bge-small-en-v1.5``
  - Small (33M parameters), fast inference
  - Strong semantic similarity performance on short texts
  - No instruction prefix needed (unlike older INSTRUCTOR models)
  - normalize_embeddings=True enables cosine similarity via dot product

Install requirements: ``uv pip install ai4data[metadata]``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from .schemas import DictionaryVariable

if TYPE_CHECKING:
    import numpy as np

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_BATCH_SIZE = 64


def _get_device() -> str:
    """Detect best available device: cuda > mps > cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _get_sentence_transformer(model_name: str, device: str):
    """Lazily load a SentenceTransformer model with a helpful install hint."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding. "
            "Install with: uv pip install ai4data[metadata]"
        ) from exc
    return SentenceTransformer(model_name, device=device)


class EmbeddingEncoder:
    """Encodes data dictionary variable texts into dense embedding vectors.

    The model is loaded lazily on the first call to ``encode()``, so
    instantiating this class has no heavy import cost.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or local path. Defaults to
        ``BAAI/bge-small-en-v1.5``.
    device : str, optional
        Device for inference (``"cuda"``, ``"mps"``, ``"cpu"``). Auto-detected
        if not specified.
    batch_size : int
        Number of texts per encoding batch. Larger batches are faster on GPU.

    Examples
    --------
    >>> encoder = EmbeddingEncoder()
    >>> texts = [encoder.build_text(v) for v in variables]
    >>> embeddings = encoder.encode(texts)  # shape (N, D)
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.device = device or _get_device()
        self.batch_size = batch_size
        self._model = None  # loaded lazily

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = _get_sentence_transformer(self.model_name, self.device)

    def encode(
        self,
        texts: List[str],
        *,
        show_progress_bar: bool = False,
    ) -> "np.ndarray":
        """Encode a list of texts into a (N, D) float32 embedding matrix.

        Embeddings are L2-normalized so that cosine similarity equals dot product.

        Parameters
        ----------
        texts : list of str
            Input texts to encode. Typically built via ``build_text()``.
        show_progress_bar : bool
            Display a tqdm progress bar during encoding.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(len(texts), embedding_dim)``.
        """
        import numpy as np

        self._ensure_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def build_text(self, variable: DictionaryVariable) -> str:
        """Construct the text representation of a variable for embedding.

        Concatenates the label and (if present) the description, separated by
        a period. This gives the model enough context to distinguish variables
        with similar short labels.

        Parameters
        ----------
        variable : DictionaryVariable
            Input variable.

        Returns
        -------
        str
            Combined text for embedding.
        """
        parts = [variable.label.strip()]
        if variable.description:
            desc = variable.description.strip()
            if desc:
                parts.append(desc)
        return ". ".join(parts)

    def encode_variables(
        self,
        variables: List[DictionaryVariable],
        *,
        show_progress_bar: bool = False,
    ) -> "np.ndarray":
        """Convenience wrapper: build texts and encode in one call.

        Parameters
        ----------
        variables : list of DictionaryVariable
            Variables to encode.
        show_progress_bar : bool
            Display a tqdm progress bar during encoding.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(len(variables), embedding_dim)``.
        """
        texts = [self.build_text(v) for v in variables]
        return self.encode(texts, show_progress_bar=show_progress_bar)
