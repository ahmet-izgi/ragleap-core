"""answer["sources"] lists each document once, in the order its first chunk was sent to the model.
It used to be built from a set, so the order changed from run to run."""

FIRST_APPEARANCE = [f"doc-{i:02d}.txt" for i in (7, 2, 11, 0, 5, 9, 1, 10, 3, 8, 4, 6)]


def _chunks():
    order = FIRST_APPEARANCE + [FIRST_APPEARANCE[3], FIRST_APPEARANCE[0], FIRST_APPEARANCE[7]]
    return [
        {"document_name": name, "text": f"text of {name} {k}", "chunk_index": k, "chunk_id": k, "document_id": name}
        for k, name in enumerate(order)
    ]


def test_sources_follow_the_order_of_first_appearance_and_have_no_repeats(rag):
    result = rag._generator.generate_answer("a question", _chunks())
    assert result["sources"] == FIRST_APPEARANCE
    assert result["sources"] != sorted(FIRST_APPEARANCE)


def test_unnamed_chunks_count_once_as_unknown_in_their_place(rag):
    chunks = [{"text": "t1"}, {"document_name": "b.txt", "text": "t2"}, {"text": "t3"}, {"document_name": "a.txt", "text": "t4"}]
    result = rag._generator.generate_answer("q", chunks)
    assert result["sources"] == ["unknown", "b.txt", "a.txt"]
