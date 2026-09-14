from types import SimpleNamespace

import src.graph.graph_builder as gb


def test_build_graph_from_chunks_reports_zero_node_extract_as_failed(monkeypatch):
    """Zero-node graph extraction should be surfaced as a failed chunk instead of silently passing."""

    class DummyGraph:
        def add_graph_documents(self, graph_documents):
            return None

    monkeypatch.setattr(gb, "get_graph", lambda: DummyGraph())
    monkeypatch.setattr(
        gb,
        "transformer",
        SimpleNamespace(
            convert_to_graph_documents=lambda chunks: [
                SimpleNamespace(nodes=[], relationships=[])
            ]
        ),
    )

    failed = gb.build_graph_from_chunks(["example text"], batch_size=999)

    assert len(failed) == 1
    assert failed[0][0] == 0
