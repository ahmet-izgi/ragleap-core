"""Tests for the evaluation additions: ranking metrics, percentile summaries, per-case latency
and token usage, the injectable support checker and compare_runs(). The pure helpers need no
database of their own (the shared conftest still opens the test database); the last test runs
the real RagLeap.evaluate() path."""
import json

import pytest

from ragleap import evaluation as ev


class FakeRag:
    """Stands in for RagLeap: answers come from a dict and every call is recorded."""

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def ask(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return dict(self.answers[query])


def ans(sources, text="answer", citations=None, usage=None):
    return {"answer": text, "sources": sources, "citations": citations or [], "usage": usage}


# ---- ranking metrics ----

def test_default_cutoffs():
    assert ev.DEFAULT_METRIC_KS == (1, 3, 5, 10)


def test_retrieval_metrics_known_values():
    m = ev.retrieval_metrics(["a", "b", "c", "d"], ["b", "d"], ks=(1, 2, 4))
    assert m["reciprocal_rank"] == 0.5
    assert (m["hit@1"], m["hit@2"], m["hit@4"]) == (0.0, 1.0, 1.0)
    assert (m["recall@1"], m["recall@2"], m["recall@4"]) == (0.0, 0.5, 1.0)
    assert (m["precision@2"], m["precision@4"]) == (0.5, 0.5)
    assert m["ndcg@1"] == 0.0
    assert m["ndcg@2"] == pytest.approx(0.38685, abs=1e-4)
    assert m["ndcg@4"] == pytest.approx(0.65092, abs=1e-4)


def test_retrieval_metrics_ideal_dcg_is_limited_by_the_cutoff():
    m = ev.retrieval_metrics(["b", "a"], ["b", "d"], ks=(1,))
    assert m["ndcg@1"] == pytest.approx(1.0)


def test_retrieval_metrics_perfect_and_empty_rankings():
    perfect = ev.retrieval_metrics(["a", "b"], ["a", "b"], ks=(2,))
    assert perfect["ndcg@2"] == pytest.approx(1.0)
    assert perfect["recall@2"] == 1.0 and perfect["precision@2"] == 1.0
    assert perfect["reciprocal_rank"] == 1.0
    empty = ev.retrieval_metrics([], ["a"], ks=(3,))
    assert empty == {"reciprocal_rank": 0.0, "hit@3": 0.0, "recall@3": 0.0, "precision@3": 0.0, "ndcg@3": 0.0}


def test_retrieval_metrics_precision_divides_by_the_cutoff_even_for_short_rankings():
    m = ev.retrieval_metrics(["a"], ["a"], ks=(5,))
    assert m["precision@5"] == pytest.approx(0.2)
    assert m["recall@5"] == 1.0


def test_retrieval_metrics_counts_a_repeated_document_once_at_its_first_position():
    m = ev.retrieval_metrics(["a", "a", "b"], ["b"], ks=(3,))
    assert m["reciprocal_rank"] == 0.5
    assert m["precision@3"] == pytest.approx(1 / 3)


def test_retrieval_metrics_recall_divides_by_the_number_of_relevant_documents():
    m = ev.retrieval_metrics(["a", "x"], ["a", "y", "z"], ks=(2,))
    assert m["recall@2"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("bad_ks", [(0,), (-1,), (True,), (1.5,), ()])
def test_retrieval_metrics_rejects_bad_cutoffs(bad_ks):
    with pytest.raises(ValueError):
        ev.retrieval_metrics(["a"], ["a"], ks=bad_ks)


def test_retrieval_metrics_needs_a_relevant_document():
    with pytest.raises(ValueError):
        ev.retrieval_metrics(["a"], [])


# ---- percentile and summaries ----

def test_percentile_nearest_rank():
    values = list(range(1, 11))
    assert ev.percentile(values, 10) == 1
    assert ev.percentile(values, 50) == 5
    assert ev.percentile(values, 95) == 10
    assert ev.percentile(values, 100) == 10
    assert ev.percentile(values, 0) == 1
    assert ev.percentile([3, 1, 2], 50) == 2
    assert ev.percentile(list(range(1, 31)), 10) == 3
    assert ev.percentile([7], 95) == 7


def test_percentile_rejects_bad_input():
    with pytest.raises(ValueError):
        ev.percentile([], 50)
    with pytest.raises(ValueError):
        ev.percentile([1], -1)
    with pytest.raises(ValueError):
        ev.percentile([1], 101)


def test_summarize_skips_none_and_reports_the_tail():
    s = ev.summarize([1.0, None, 0.0, 0.5, 0.25], fail_below=0.5)
    assert s["n"] == 4
    assert s["mean"] == pytest.approx(0.4375)
    assert (s["min"], s["p10"], s["p50"], s["p95"]) == (0.0, 0.0, 0.25, 1.0)
    assert s["fail_rate"] == pytest.approx(0.5)


def test_summarize_of_nothing_is_none_and_without_threshold_has_no_fail_rate():
    assert ev.summarize([]) is None
    assert ev.summarize([None, None]) is None
    assert "fail_rate" not in ev.summarize([1.0])


# ---- per-case results ----

def test_evaluate_case_records_latency_usage_and_metrics():
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    rag = FakeRag({"q": ans(["a", "b"], usage=usage)})
    r = ev.evaluate_case(rag, {"query": "q", "relevant_documents": ["b"]}, metric_ks=(1, 2))
    assert 0 <= r["latency_s"] < 5
    assert r["usage"] == usage
    assert set(r["metrics"]) == {
        "reciprocal_rank", "hit@1", "hit@2", "recall@1", "recall@2",
        "precision@1", "precision@2", "ndcg@1", "ndcg@2",
    }
    assert r["metrics"]["hit@1"] == 0.0 and r["metrics"]["hit@2"] == 1.0
    assert r["support"] is None and r["support_error"] is None


def test_evaluate_case_without_relevant_documents_has_no_metrics():
    r = ev.evaluate_case(FakeRag({"q": ans(["a"])}), {"query": "q"})
    assert r["metrics"] is None


def test_support_checker_receives_the_answer_text_and_the_citations():
    seen = []

    def checker(text, citations):
        seen.append((text, citations))
        return 0.75

    cites = [{"text_preview": "t"}]
    r = ev.evaluate_case(FakeRag({"q": ans(["a"], text="the answer", citations=cites)}), {"query": "q"}, support_checker=checker)
    assert r["support"] == 0.75 and r["support_error"] is None
    assert seen == [("the answer", cites)]


def test_support_checker_accepts_the_boundary_values():
    for value in (0, 1):
        r = ev.evaluate_case(FakeRag({"q": ans(["a"])}), {"query": "q"}, support_checker=lambda text, citations: value)
        assert r["support"] == float(value) and r["support_error"] is None


def test_support_checker_failures_are_recorded_not_raised():
    def boom(text, citations):
        raise RuntimeError("judge down")

    r = ev.evaluate_case(FakeRag({"q": ans(["a"])}), {"query": "q"}, support_checker=boom)
    assert r["support"] is None
    assert r["support_error"] == "RuntimeError: judge down"


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan"), True, "high", None])
def test_support_checker_must_return_a_number_between_0_and_1(bad):
    r = ev.evaluate_case(FakeRag({"q": ans(["a"])}), {"query": "q"}, support_checker=lambda text, citations: bad)
    assert r["support"] is None
    assert r["support_error"].startswith("support_checker must return a number between 0 and 1")


# ---- whole runs ----

def test_evaluate_summary_reports_mean_tail_and_fail_rate():
    usage_1 = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    usage_2 = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    answers = {
        "q1": ans(["a", "b"], text="s1", usage=usage_1),
        "q2": ans(["b", "a"], text="s2", usage=usage_2),
        "q3": ans(["c"], text="s3", usage=None),
    }
    cases = [
        {"query": "q1", "relevant_documents": ["a"]},
        {"query": "q2", "relevant_documents": ["x"]},
        {"query": "q3", "relevant_documents": ["c"]},
    ]
    scores = {"s1": 1.0, "s2": 0.2, "s3": 0.9}
    rag = FakeRag(answers)
    result = ev.evaluate(rag, cases, metric_ks=(1,), support_checker=lambda text, cites: scores[text], fail_below=0.95, top_k=3)
    s = result["summary"]
    assert s["cases"] == 3 and s["fail_below"] == 0.95 and s["support_errors"] == 0
    assert s["retrieval"]["recall@1"]["mean"] == pytest.approx(2 / 3)
    assert s["retrieval"]["recall@1"]["min"] == 0.0
    assert "fail_rate" not in s["retrieval"]["recall@1"]
    support = s["support"]
    assert (support["n"], support["min"], support["p10"], support["p50"], support["p95"]) == (3, 0.2, 0.2, 0.9, 1.0)
    assert support["mean"] == pytest.approx(0.7)
    assert support["fail_rate"] == pytest.approx(2 / 3)
    assert s["usage_totals"] == {"prompt_tokens": 17, "completion_tokens": 8, "total_tokens": 25}
    assert s["latency_s"]["n"] == 3
    assert s["keyword_coverage"] is None and s["groundedness"] is None
    assert rag.calls == [("q1", {"top_k": 3}), ("q2", {"top_k": 3}), ("q3", {"top_k": 3})]
    json.dumps(result)


def test_evaluate_counts_support_errors_and_keeps_going():
    answers = {"q1": ans(["a"], text="ok"), "q2": ans(["a"], text="bad")}

    def checker(text, citations):
        if text == "bad":
            raise RuntimeError("judge down")
        return 1.0

    result = ev.evaluate(FakeRag(answers), [{"query": "q1"}, {"query": "q2"}], support_checker=checker)
    assert result["summary"]["support_errors"] == 1
    assert result["summary"]["support"]["n"] == 1
    assert result["results"][1]["support_error"] == "RuntimeError: judge down"


def test_evaluate_validates_its_new_options():
    rag = FakeRag({"q": ans(["a"])})
    with pytest.raises(ValueError):
        ev.evaluate(rag, [{"query": "q"}], metric_ks=(0,))
    with pytest.raises(ValueError):
        ev.evaluate(rag, [{"query": "q"}], fail_below=1.5)
    with pytest.raises(ValueError):
        ev.evaluate(rag, [])


def test_evaluate_keeps_the_original_result_shape_and_adds_a_summary():
    rag = FakeRag({"q": ans(["a"], text="has bananas", citations=[{"text_preview": "bananas here"}])})
    result = ev.evaluate(rag, [{"query": "q", "expected_document": "a", "expected_keywords": ["bananas"]}])
    assert result["retrieval_hit_rate"] == 1.0
    assert result["keyword_coverage_rate"] == 1.0
    assert result["groundedness_rate"] == 1.0
    assert result["results"][0]["keywords_found"] == ["bananas"]
    assert result["results"][0]["metrics"] is None
    assert result["summary"]["retrieval"] == {}


# ---- comparing runs ----

CASES = [
    {"query": "q1", "expected_document": "a", "relevant_documents": ["a"]},
    {"query": "q2", "expected_document": "b", "relevant_documents": ["b"]},
]
GOOD = {"q1": ans(["a", "b"]), "q2": ans(["b", "a"])}
WORSE = {"q1": ans(["b", "a"]), "q2": ans(["b", "a"])}


def run(answers, cases=CASES, **kwargs):
    return ev.evaluate(FakeRag(answers), cases, metric_ks=(1,), **kwargs)


def test_compare_flags_a_ranking_regression_even_when_the_hit_rate_is_unchanged():
    cmp = ev.compare_runs(run(GOOD), run(WORSE))
    assert cmp["metrics"]["retrieval_hit_rate"]["delta"] == 0.0
    assert "retrieval_hit_rate" not in cmp["regressions"]
    assert cmp["metrics"]["retrieval.recall@1"] == {"baseline": 1.0, "candidate": 0.5, "delta": -0.5}
    assert "retrieval.recall@1" in cmp["regressions"]
    assert cmp["has_regressions"] is True
    assert cmp["cases"] == 2
    assert {"query": "q1", "metric": "retrieval.recall@1", "kind": "regression", "baseline": 1.0, "candidate": 0.0} in cmp["case_changes"]
    json.dumps(cmp)


def test_compare_identical_runs_have_no_changes():
    cmp = ev.compare_runs(run(GOOD), run(GOOD))
    assert cmp["regressions"] == [] and cmp["improvements"] == [] and cmp["case_changes"] == []
    assert cmp["has_regressions"] is False


def test_compare_reports_an_improvement():
    cmp = ev.compare_runs(run(WORSE), run(GOOD))
    assert "retrieval.recall@1" in cmp["improvements"]
    assert cmp["regressions"] == [] and cmp["has_regressions"] is False


def test_compare_tolerance_is_inclusive_for_aggregates_but_cases_still_count():
    cmp = ev.compare_runs(run(GOOD), run(WORSE), tolerance=0.5)
    assert cmp["regressions"] == []
    assert cmp["tolerance"] == 0.5
    assert cmp["has_regressions"] is True


def test_compare_ignores_float_noise():
    base = {"retrieval_hit_rate": 0.30000000000000004, "results": []}
    cand = {"retrieval_hit_rate": 0.3, "results": []}
    cmp = ev.compare_runs(base, cand)
    assert cmp["regressions"] == [] and cmp["improvements"] == []
    assert cmp["has_regressions"] is False
    assert cmp["metrics"]["retrieval_hit_rate"]["delta"] == 0.0
    assert cmp["latency_s"] is None


def test_compare_refuses_runs_that_are_not_comparable():
    with pytest.raises(ValueError, match="cases against"):
        ev.compare_runs(run(GOOD), run({"q1": ans(["a"])}, CASES[:1]))
    other = [dict(CASES[0]), {"query": "q3", "expected_document": "b"}]
    with pytest.raises(ValueError, match="different query"):
        ev.compare_runs(run(GOOD), run({"q1": ans(["a"]), "q3": ans(["b"])}, other))
    with pytest.raises(ValueError, match="negative"):
        ev.compare_runs(run(GOOD), run(GOOD), tolerance=-0.1)


def test_compare_reports_cases_that_flip_even_when_the_hit_rate_is_unchanged():
    cases = [{"query": "q1", "expected_document": "a"}, {"query": "q2", "expected_document": "b"}]
    base = ev.evaluate(FakeRag({"q1": ans(["a"]), "q2": ans(["x"])}), cases)
    cand = ev.evaluate(FakeRag({"q1": ans(["x"]), "q2": ans(["b"])}), cases)
    cmp = ev.compare_runs(base, cand)
    assert cmp["metrics"]["retrieval_hit_rate"]["delta"] == 0.0
    assert cmp["regressions"] == [] and cmp["improvements"] == []
    assert {"query": "q1", "metric": "retrieval_hit", "kind": "regression", "baseline": True, "candidate": False} in cmp["case_changes"]
    assert {"query": "q2", "metric": "retrieval_hit", "kind": "improvement", "baseline": False, "candidate": True} in cmp["case_changes"]
    assert cmp["has_regressions"] is True
    json.dumps(cmp)


def test_compare_uses_the_support_mean():
    cases = [{"query": "q1"}, {"query": "q2"}]
    answers = {"q1": ans(["a"], text="s1"), "q2": ans(["a"], text="s2")}
    base = ev.evaluate(FakeRag(answers), cases, support_checker=lambda t, c: {"s1": 1.0, "s2": 0.2}[t])
    cand = ev.evaluate(FakeRag(answers), cases, support_checker=lambda t, c: {"s1": 1.0, "s2": 0.9}[t])
    cmp = ev.compare_runs(base, cand)
    assert cmp["metrics"]["support"]["baseline"] == pytest.approx(0.6)
    assert cmp["metrics"]["support"]["candidate"] == pytest.approx(0.95)
    assert "support" in cmp["improvements"]
    assert cmp["has_regressions"] is False


def test_compare_reports_latency_when_both_runs_have_it():
    def stats(mean, p95):
        return {"summary": {"latency_s": {"mean": mean, "p95": p95}}, "results": []}

    cmp = ev.compare_runs(stats(1.0, 2.0), stats(1.5, 3.0))
    assert cmp["latency_s"] == {"baseline_mean": 1.0, "candidate_mean": 1.5, "baseline_p95": 2.0, "candidate_p95": 3.0}
    assert cmp["regressions"] == []


# ---- ranking order ----

def test_ranking_follows_the_citations_not_the_unordered_sources():
    citations = [
        {"document_name": "a", "text_preview": "x"},
        {"document_name": "a", "text_preview": "y"},
        {"document_name": "z", "text_preview": "w"},
    ]
    rag = FakeRag({"q": ans(["z", "a"], citations=citations)})
    r = ev.evaluate_case(rag, {"query": "q", "relevant_documents": ["z"]}, metric_ks=(1, 2))
    assert r["ranked_documents"] == ["a", "z"]
    assert r["metrics"]["reciprocal_rank"] == 0.5
    assert r["metrics"]["hit@1"] == 0.0 and r["metrics"]["hit@2"] == 1.0


def test_ranking_falls_back_to_the_sources_when_no_citation_has_a_document_name():
    rag = FakeRag({"q": ans(["b", "a"], citations=[{"text_preview": "t"}])})
    r = ev.evaluate_case(rag, {"query": "q", "relevant_documents": ["a"]}, metric_ks=(1,))
    assert r["ranked_documents"] == ["b", "a"]
    assert r["metrics"]["reciprocal_rank"] == 0.5


# ---- the real RagLeap.evaluate() path (needs the test database) ----

def test_rag_evaluate_reports_ranking_metrics_latency_and_summary(rag):
    rag.ingest_text("bananas.txt", "This document is about bananas and tropical fruit.")
    rag.ingest_text("spaceships.txt", "This document is about spaceships and rockets.")
    result = rag.evaluate(
        [{"query": "bananas", "expected_document": "bananas.txt", "relevant_documents": ["bananas.txt"]}],
        hybrid=True,
        metric_ks=(5,),
    )
    case = result["results"][0]
    assert case["metrics"]["hit@5"] == 1.0
    assert case["latency_s"] >= 0
    assert case["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert result["summary"]["usage_totals"]["total_tokens"] == 15
    assert result["summary"]["latency_s"]["n"] == 1
    assert set(case["ranked_documents"]) == set(case["sources"])
    assert "bananas.txt" in case["ranked_documents"]
