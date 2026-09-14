import numpy as np

from nanorag.embeddings.base import Embedder
from tests.fakes import FakeEmbedder


def test_fake_embedder_satisfies_the_embedder_protocol():
    assert isinstance(FakeEmbedder(), Embedder)


def test_an_object_missing_embed_query_does_not_satisfy_the_protocol():
    class Incomplete:
        dim = 8
        model_id = "incomplete"

        def embed(self, texts):
            return np.zeros((len(texts), self.dim), dtype=np.float32)

    assert not isinstance(Incomplete(), Embedder)


def test_an_object_missing_dim_does_not_satisfy_the_protocol():
    class Incomplete:
        model_id = "incomplete"

        def embed(self, texts):
            return np.zeros((len(texts), 8), dtype=np.float32)

        def embed_query(self, text):
            return np.zeros(8, dtype=np.float32)

    assert not isinstance(Incomplete(), Embedder)
