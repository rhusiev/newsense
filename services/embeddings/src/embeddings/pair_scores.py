"""Local same-storyline scoring for story pairs.

The optional model is a quantized ONNX cross-encoder. It reads two story texts
and returns the probability that they belong to one storyline. Loading is
lazy so the worker starts without a model when THREAD_PAIR_MODEL is unset.
"""
import logging
import os

import numpy as np

from .config import (
    THREAD_PAIR_BATCH,
    THREAD_PAIR_MAX_LEN,
    THREAD_PAIR_MODEL,
    THREAD_PAIR_THRESHOLD,
)

logger = logging.getLogger("thread-pairs")

_session = None
_tokenizer = None


def enabled() -> bool:
    return bool(THREAD_PAIR_MODEL)


def _load() -> None:
    global _session, _tokenizer
    if _session is not None:
        return
    import onnxruntime as ort
    from transformers import AutoTokenizer

    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    onnx_path = THREAD_PAIR_MODEL
    if os.path.isdir(onnx_path):
        onnx_path = os.path.join(onnx_path, "model.onnx")
    _session = ort.InferenceSession(
        onnx_path, options, providers=["CPUExecutionProvider"]
    )
    _tokenizer = AutoTokenizer.from_pretrained(THREAD_PAIR_MODEL)
    logger.info("Loaded thread pair scorer %s", THREAD_PAIR_MODEL)


def score(texts_a: list[str], texts_b: list[str]) -> np.ndarray:
    """Return one probability per pair, or an empty array when disabled."""
    if not enabled():
        return np.zeros(0, dtype=np.float32)
    _load()
    output = []
    for start in range(0, len(texts_a), THREAD_PAIR_BATCH):
        stop = start + THREAD_PAIR_BATCH
        encoded = _tokenizer(
            texts_a[start:stop],
            texts_b[start:stop],
            truncation=True,
            max_length=THREAD_PAIR_MAX_LEN,
            padding=True,
            return_tensors="np",
        )
        feed = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        logits = _session.run(["logits"], feed)[0]
        output.extend(1 / (1 + np.exp(-logits[:, 1].astype(np.float32))))
    return np.asarray(output, dtype=np.float32)


def accept(probability: float) -> bool:
    return probability >= THREAD_PAIR_THRESHOLD
