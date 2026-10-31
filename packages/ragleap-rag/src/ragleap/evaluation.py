"""
Lightweight evaluation utility for ragleap-rag.

Honest scope: this is NOT an LLM-as-judge evaluation framework (like
Ragas's faithfulness/relevancy scoring) - those require careful judge-
prompt design and calibration, and belong in a separate, dedicated
tool (planned as part of ragleap-observability, which will span more
than just RAG). This module does these deterministic, measurable
checks instead:

1. Retrieval hit rate - did the expected source document actually
   appear in the retrieved chunks for a query?
2. Keyword coverage - what fraction of expected keywords appear
   (case-insensitive substring match) in the generated answer?
3. Citation groundedness - of the keywords found in the answer, what
   fraction also appear in the text of the chunks the answer actually
   cited? A low score here suggests the model may be answering from
   outside the retrieved context (a real hallucination signal, though
   still a heuristic - substring matching is not semantic understanding).
4. Ranking metrics - for cases that list relevant_documents: hit@k,
   recall@k, precision@k, reciprocal rank and nDCG@k over the
   document-level ranking of the chunks sent to the model (after the
   context-size trim), read from answer["citations"] in chunk order.
   answer["sources"] is built from a set and has no stable order, so
   it is only used when no citation carries a document name. Binary
   relevance; a document that appears several times counts once, at
   its first position; precision@k always divides by k.

Every run also records per-case latency and token usage and reports the
tail next to the mean (min, p10, p50 and p95 by the nearest-rank method,
plus a fail rate below a threshold you choose), because a mean hides
rare failures. Pass support_checker=callable(answer_text, citations)
-> float between 0 and 1 to score how well each answer is supported by
what it cited, with your own NLI model or LLM judge; this package ships
no judge. compare_runs() diffs two runs of the same cases and flags
regressions.

The checks need no LLM call themselves, so running an evaluation is
fast and free beyond the ask() calls it makes. Latency is the only
non-deterministic number.
"""
import logging
import math
import time
from fractions import Fraction
from typing import Callable, Collection, Dict, List, Optional, Sequence, TypedDict

logger = logging.getLogger(__name__)

DEFAULT_METRIC_KS = (1, 3, 5, 10)
DEFAULT_FAIL_BELOW = 0.5


class EvalCase(TypedDict, total=False):
    query: str
    expected_document: Optional[str]
    expected_keywords: Optional[List[str]]
    relevant_documents: Optional[List[str]]


# (answer_text, citations) -> a score between 0 and 1
SupportChecker = Callable[[str, List[Dict]], float]


def _keyword_hits(text: str, keywords: List[str]) -> List[str]:
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


def _dedupe_keep_order(items):
    seen = set()
    ordered = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _validate_ks(ks: Sequence[int]) -> None:
    if not ks:
        raise ValueError("metric_ks must contain at least one cutoff.")
    for k in ks:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f"metric_ks must contain positive integers, got {k!r}")


def _ranked_documents(answer: Dict) -> List[str]:
    """
    Document names in rank order, from answer["citations"] (one entry per
    chunk sent to the model, in retrieval or rerank order). answer["sources"]
    is built from a set and has no stable order, so it is only the fallback
    when no citation carries a document name.
    """
    names = [c.get("document_name") for c in answer.get("citations") or [] if c.get("document_name")]
    if names:
        return _dedupe_keep_order(names)
    return list(answer.get("sources") or [])


def retrieval_metrics(
    ranked: Sequence[str],
    relevant: Collection[str],
    ks: Sequence[int] = DEFAULT_METRIC_KS,
) -> Dict[str, float]:
    """
    Ranking metrics for one query. ranked: document names, best first (a
    document that appears several times counts once, at its first
    position). relevant: the document names that count as relevant
    (binary relevance). For each cutoff k: hit@k, recall@k, precision@k
    (always divided by k, even when fewer documents came back) and
    nDCG@k, plus reciprocal_rank.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError("retrieval_metrics() requires at least one relevant document.")
    _validate_ks(ks)
    ranking = _dedupe_keep_order(ranked)
    gains = [1 if doc in relevant_set else 0 for doc in ranking]
    first = next((i for i, g in enumerate(gains) if g), None)
    metrics: Dict[str, float] = {
        "reciprocal_rank": 1.0 / (first + 1) if first is not None else 0.0,
    }
    for k in ks:
        top = gains[:k]
        found = sum(top)
        metrics[f"hit@{k}"] = 1.0 if found else 0.0
        metrics[f"recall@{k}"] = found / len(relevant_set)
        metrics[f"precision@{k}"] = found / k
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(top))
        ideal = min(len(relevant_set), k)
        idcg = sum(1 / math.log2(i + 2) for i in range(ideal))
        metrics[f"ndcg@{k}"] = dcg / idcg
    return metrics


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile (q from 0 to 100); the smallest value for q = 0."""
    if not values:
        raise ValueError("percentile() requires at least one value.")
    if not 0 <= q <= 100:
        raise ValueError(f"q must be between 0 and 100, got {q!r}")
    ordered = sorted(values)
    rank = max(1, math.ceil(Fraction(str(q)) * len(ordered) / 100))
    return ordered[rank - 1]


def summarize(values: Sequence[Optional[float]], fail_below: Optional[float] = None) -> Optional[Dict[str, float]]:
    """
    Mean and tail of a list of scores (None entries are skipped): n, mean,
    min, p10, p50, p95 and, if fail_below is given, fail_rate - the share
    of scores below it. None if there is nothing to summarize.
    """
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    stats: Dict[str, float] = {
        "n": len(nums),
        "mean": sum(nums) / len(nums),
        "min": min(nums),
        "p10": percentile(nums, 10),
        "p50": percentile(nums, 50),
        "p95": percentile(nums, 95),
    }
    if fail_below is not None:
        stats["fail_rate"] = sum(1 for v in nums if v < fail_below) / len(nums)
    return stats


def _run_support_checker(checker: SupportChecker, answer: Dict):
    try:
        score = checker(answer.get("answer") or "", list(answer.get("citations") or []))
    except Exception as exc:  # a broken judge must not abort a whole run
        logger.warning(f"support_checker failed: {exc}")
        return None, f"{type(exc).__name__}: {exc}"
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
        return None, f"support_checker must return a number between 0 and 1, got {score!r}"
    return float(score), None


def evaluate_case(
    rag,
    case: Dict,
    *,
    metric_ks: Sequence[int] = DEFAULT_METRIC_KS,
    support_checker: Optional[SupportChecker] = None,
    **ask_kwargs,
) -> Dict:
    """
    Run a single evaluation case through rag.ask() and score it.
    Returns per-case detail - see evaluate() for the aggregated form.
    """
    query = case["query"]
    expected_document = case.get("expected_document")
    expected_keywords = case.get("expected_keywords") or []
    relevant_documents = case.get("relevant_documents") or []

    started = time.perf_counter()
    answer = rag.ask(query, **ask_kwargs)
    latency_s = time.perf_counter() - started

    retrieval_hit = None
    if expected_document is not None:
        retrieval_hit = expected_document in answer.get("sources", [])

    keyword_hits = _keyword_hits(answer.get("answer", ""), expected_keywords)
    keyword_coverage = (len(keyword_hits) / len(expected_keywords)) if expected_keywords else None

    groundedness = None
    if keyword_hits:
        cited_text = " ".join(c.get("text_preview", "") for c in answer.get("citations", []))
        grounded_hits = _keyword_hits(cited_text, keyword_hits)
        groundedness = len(grounded_hits) / len(keyword_hits)

    ranked_documents = _ranked_documents(answer)
    metrics = None
    if relevant_documents:
        metrics = retrieval_metrics(ranked_documents, relevant_documents, metric_ks)

    support = None
    support_error = None
    if support_checker is not None:
        support, support_error = _run_support_checker(support_checker, answer)

    return {
        "query": query,
        "answer": answer.get("answer"),
        "sources": answer.get("sources", []),
        "ranked_documents": ranked_documents,
        "retrieval_hit": retrieval_hit,
        "keyword_coverage": keyword_coverage,
        "keywords_found": keyword_hits,
        "groundedness": groundedness,
        "metrics": metrics,
        "support": support,
        "support_error": support_error,
        "latency_s": latency_s,
        "usage": answer.get("usage"),
    }


def _summarize_run(results: List[Dict], fail_below: float) -> Dict:
    metric_names: List[str] = []
    for result in results:
        for name in result.get("metrics") or {}:
            if name not in metric_names:
                metric_names.append(name)
    retrieval = {
        name: summarize([(r.get("metrics") or {}).get(name) for r in results])
        for name in metric_names
    }
    usage_totals: Dict[str, float] = {}
    for result in results:
        usage = result.get("usage")
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    usage_totals[key] = usage_totals.get(key, 0) + value
    return {
        "cases": len(results),
        "fail_below": fail_below,
        "latency_s": summarize([r["latency_s"] for r in results]),
        "keyword_coverage": summarize([r["keyword_coverage"] for r in results], fail_below),
        "groundedness": summarize([r["groundedness"] for r in results], fail_below),
        "support": summarize([r["support"] for r in results], fail_below),
        "retrieval": retrieval,
        "usage_totals": usage_totals or None,
        "support_errors": sum(1 for r in results if r.get("support_error")),
    }


def evaluate(
    rag,
    test_cases: List[Dict],
    *,
    metric_ks: Sequence[int] = DEFAULT_METRIC_KS,
    support_checker: Optional[SupportChecker] = None,
    fail_below: float = DEFAULT_FAIL_BELOW,
    **ask_kwargs,
) -> Dict:
    """
    Run a labeled test set through rag.ask() and report aggregate
    deterministic quality signals - retrieval hit rate, keyword
    coverage, and citation groundedness - plus ranking metrics,
    latency, token usage and a tail-aware summary. Not a replacement
    for human review or a full LLM-as-judge eval framework - a fast,
    free, repeatable sanity check for catching regressions in your own
    retrieval/generation setup.

    test_cases: list of dicts, each with:
      - "query": str, required
      - "expected_document": str, optional - a document_name that
        should appear in answer["sources"] for this query
      - "expected_keywords": List[str], optional - substrings that
        should appear (case-insensitive) in the generated answer
      - "relevant_documents": List[str], optional - document names that
        count as relevant, for the ranking metrics (hit@k, recall@k,
        precision@k, reciprocal rank, nDCG@k), measured on the chunks sent to
        the model in citation order

    Keyword-only options (they are not passed to ask()):
      - metric_ks: cutoffs for the ranking metrics, default (1, 3, 5, 10)
      - support_checker: optional callable(answer_text, citations) ->
        float between 0 and 1; its failures are recorded per case
      - fail_below: scores below this count towards the fail rate in the
        summary, default 0.5

    Returns: {
        "retrieval_hit_rate": float | None,   # None if no cases had expected_document
        "keyword_coverage_rate": float | None, # None if no cases had expected_keywords
        "groundedness_rate": float | None,     # None if no keyword hits occurred anywhere
        "results": [per-case detail dicts, see evaluate_case()],
        "summary": {cases, fail_below, latency_s, keyword_coverage,
                    groundedness, support, retrieval, usage_totals,
                    support_errors},  # each stats entry has n, mean, min, p10, p50, p95
    }
    """
    if not test_cases:
        raise ValueError("evaluate() requires at least one test case.")
    _validate_ks(metric_ks)
    if not 0 <= fail_below <= 1:
        raise ValueError(f"fail_below must be between 0 and 1, got {fail_below!r}")

    results = [
        evaluate_case(rag, case, metric_ks=metric_ks, support_checker=support_checker, **ask_kwargs)
        for case in test_cases
    ]

    hit_results = [r["retrieval_hit"] for r in results if r["retrieval_hit"] is not None]
    retrieval_hit_rate = (sum(hit_results) / len(hit_results)) if hit_results else None

    coverage_results = [r["keyword_coverage"] for r in results if r["keyword_coverage"] is not None]
    keyword_coverage_rate = (sum(coverage_results) / len(coverage_results)) if coverage_results else None

    groundedness_results = [r["groundedness"] for r in results if r["groundedness"] is not None]
    groundedness_rate = (sum(groundedness_results) / len(groundedness_results)) if groundedness_results else None

    logger.info(
        f"Evaluation complete: {len(results)} cases, "
        f"retrieval_hit_rate={retrieval_hit_rate}, "
        f"keyword_coverage_rate={keyword_coverage_rate}, "
        f"groundedness_rate={groundedness_rate}"
    )

    return {
        "retrieval_hit_rate": retrieval_hit_rate,
        "keyword_coverage_rate": keyword_coverage_rate,
        "groundedness_rate": groundedness_rate,
        "results": results,
        "summary": _summarize_run(results, fail_below),
    }


_TOP_LEVEL_SCORES = ("retrieval_hit_rate", "keyword_coverage_rate", "groundedness_rate")


def _direction(delta: float, tolerance: float) -> Optional[str]:
    if delta < -tolerance:
        return "regression"
    if delta > tolerance:
        return "improvement"
    return None


def _run_scores(run: Dict) -> Dict[str, Optional[float]]:
    scores: Dict[str, Optional[float]] = {name: run.get(name) for name in _TOP_LEVEL_SCORES}
    summary = run.get("summary") or {}
    for metric, stats in (summary.get("retrieval") or {}).items():
        scores[f"retrieval.{metric}"] = stats["mean"]
    support = summary.get("support")
    if support:
        scores["support"] = support["mean"]
    return scores


def _case_scores(case: Dict) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for name in ("keyword_coverage", "groundedness", "support"):
        if case.get(name) is not None:
            scores[name] = case[name]
    for name, value in (case.get("metrics") or {}).items():
        scores[f"retrieval.{name}"] = value
    return scores


def _compare_latency(baseline: Dict, candidate: Dict) -> Optional[Dict]:
    b = (baseline.get("summary") or {}).get("latency_s")
    c = (candidate.get("summary") or {}).get("latency_s")
    if not b or not c:
        return None
    return {
        "baseline_mean": b["mean"],
        "candidate_mean": c["mean"],
        "baseline_p95": b["p95"],
        "candidate_p95": c["p95"],
    }


def compare_runs(baseline: Dict, candidate: Dict, tolerance: float = 0.0) -> Dict:
    """
    Compare two evaluate() results of the same cases (same queries, same
    order). A score counts as a regression when it dropped by more than
    tolerance (an improvement when it rose by more); differences below
    1e-12 are ignored as float noise. Aggregate scores are compared, and
    every case separately, so a case that flipped is not hidden by
    another one that improved. Latency is reported, never flagged.

    Returns: {cases, tolerance, metrics: {name: {baseline, candidate,
    delta}}, regressions: [names], improvements: [names], case_changes:
    [{query, metric, kind, baseline, candidate}], latency_s, has_regressions}
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must not be negative, got {tolerance!r}")
    base_results = baseline.get("results") or []
    cand_results = candidate.get("results") or []
    if len(base_results) != len(cand_results):
        raise ValueError(
            f"runs are not comparable: {len(base_results)} cases against {len(cand_results)}"
        )
    for index, (b, c) in enumerate(zip(base_results, cand_results)):
        if b.get("query") != c.get("query"):
            raise ValueError(f"runs are not comparable: case {index} has a different query")

    base_scores = _run_scores(baseline)
    cand_scores = _run_scores(candidate)
    metrics: Dict[str, Dict] = {}
    regressions: List[str] = []
    improvements: List[str] = []
    for name in sorted(set(base_scores) | set(cand_scores)):
        b_value, c_value = base_scores.get(name), cand_scores.get(name)
        if b_value is None or c_value is None:
            metrics[name] = {"baseline": b_value, "candidate": c_value, "delta": None}
            continue
        delta = round(c_value - b_value, 12)
        metrics[name] = {"baseline": b_value, "candidate": c_value, "delta": delta}
        direction = _direction(delta, tolerance)
        if direction == "regression":
            regressions.append(name)
        elif direction == "improvement":
            improvements.append(name)

    case_changes: List[Dict] = []
    for b, c in zip(base_results, cand_results):
        query = b.get("query")
        b_hit, c_hit = b.get("retrieval_hit"), c.get("retrieval_hit")
        if b_hit is True and c_hit is False:
            case_changes.append({"query": query, "metric": "retrieval_hit", "kind": "regression", "baseline": True, "candidate": False})
        elif b_hit is False and c_hit is True:
            case_changes.append({"query": query, "metric": "retrieval_hit", "kind": "improvement", "baseline": False, "candidate": True})
        b_case, c_case = _case_scores(b), _case_scores(c)
        for name in sorted(set(b_case) & set(c_case)):
            delta = round(c_case[name] - b_case[name], 12)
            direction = _direction(delta, tolerance)
            if direction is None:
                continue
            case_changes.append({"query": query, "metric": name, "kind": direction, "baseline": b_case[name], "candidate": c_case[name]})

    has_regressions = bool(regressions) or any(change["kind"] == "regression" for change in case_changes)
    return {
        "cases": len(base_results),
        "tolerance": tolerance,
        "metrics": metrics,
        "regressions": regressions,
        "improvements": improvements,
        "case_changes": case_changes,
        "latency_s": _compare_latency(baseline, candidate),
        "has_regressions": has_regressions,
    }
