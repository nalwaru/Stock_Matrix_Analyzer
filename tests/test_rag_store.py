from types import SimpleNamespace as NS

from rag import store
from rag.chunking import Chunk


class FakeIndex:
    def __init__(self, name, pc):
        self.name, self.pc = name, pc

    def upsert_records(self, *, records, namespace):
        self.pc.upserts.append((self.name, namespace, list(records)))

    def describe_index_stats(self):
        counts = {}
        for name, ns, recs in self.pc.upserts:
            if name == self.name:
                counts[ns] = counts.get(ns, 0) + len(recs)
        return NS(namespaces={ns: NS(vector_count=n) for ns, n in counts.items()})


class FakePinecone:
    def __init__(self, existing=()):
        self.existing, self.created, self.upserts = set(existing), [], []

    def has_index(self, name):
        return name in self.existing

    def create_index_for_model(self, **kwargs):
        self.created.append(kwargs)
        self.existing.add(kwargs["name"])

    def Index(self, name):
        return FakeIndex(name, self)


def chunks(n):
    return [Chunk(f"id{i}", f"text {i}", {"ticker": "AAPL", "section": "Item 7", "chunk_type": "text"}) for i in range(n)]


def test_ensure_indexes_is_idempotent():
    pc = FakePinecone()
    store.ensure_indexes(pc)
    store.ensure_indexes(pc)
    assert {c["name"] for c in pc.created} == {store.DENSE_INDEX, store.SPARSE_INDEX}
    assert len(pc.created) == 2
    models = {c["name"]: c["embed"]["model"] for c in pc.created}
    assert models[store.DENSE_INDEX] == store.DENSE_MODEL
    assert models[store.SPARSE_INDEX] == store.SPARSE_MODEL


def test_upsert_batches_of_90_to_both_indexes():
    pc = FakePinecone()
    store.upsert_chunks(pc, "fixed", chunks(200))
    per_index = {}
    for name, ns, recs in pc.upserts:
        assert ns == "fixed" and len(recs) <= 90
        per_index.setdefault(name, []).append(len(recs))
    assert per_index == {store.DENSE_INDEX: [90, 90, 20], store.SPARSE_INDEX: [90, 90, 20]}


def test_upsert_returns_record_count_and_flattens_metadata():
    pc = FakePinecone()
    assert store.upsert_chunks(pc, "semantic", chunks(3)) == 3
    record = pc.upserts[0][2][0]
    assert record["_id"] == "id0" and record["text"] == "text 0" and record["ticker"] == "AAPL"


def test_stats_counts_records_per_index_and_namespace():
    pc = FakePinecone()
    store.upsert_chunks(pc, "fixed", chunks(5))
    assert store.stats(pc) == {f"{store.DENSE_INDEX}/fixed": 5, f"{store.SPARSE_INDEX}/fixed": 5}
