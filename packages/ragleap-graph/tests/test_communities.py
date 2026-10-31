"""Tests for ragleap_graph.communities: deterministic Louvain community detection.
Pure Python, no database. The comparisons with networkx run only where networkx is installed."""
import hashlib
import itertools
import random
from collections import Counter

import pytest

from ragleap_graph import communities as cm
from ragleap_graph.communities import detect_communities, members_hash, modularity

TWO_TRIANGLES = [
    ("a1", "a2", 1), ("a2", "a3", 1), ("a1", "a3", 1),
    ("b1", "b2", 1), ("b2", "b3", 1), ("b1", "b3", 1),
    ("a3", "b1", 1),
]
WEIGHTED_PAIRS = [("a", "b", 10), ("c", "d", 10), ("b", "c", 1)]
TIE = [("a", "b", 5), ("c", "d", 5), ("a", "m", 1), ("c", "m", 1)]


def groups(result):
    return [c["members"] for c in result]


def all_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for partition in all_partitions(rest):
        for i in range(len(partition)):
            yield partition[:i] + [[first] + partition[i]] + partition[i + 1:]
        yield [[first]] + partition


def adjusted_rand_index(truth, found):
    nodes = list(truth)
    pairs = Counter((truth[n], found[n]) for n in nodes)
    left = Counter(truth[n] for n in nodes)
    right = Counter(found[n] for n in nodes)

    def comb2(x):
        return x * (x - 1) / 2

    sum_pairs = sum(comb2(v) for v in pairs.values())
    sum_left = sum(comb2(v) for v in left.values())
    sum_right = sum(comb2(v) for v in right.values())
    expected = sum_left * sum_right / comb2(len(nodes))
    maximum = (sum_left + sum_right) / 2
    return (sum_pairs - expected) / (maximum - expected)


def planted_graph(seed=7, group_count=4, size=10, p_in=0.6, p_out=0.02):
    rng = random.Random(seed)
    nodes = [f"g{g}n{i:02d}" for g in range(group_count) for i in range(size)]
    truth = {name: name[:2] for name in nodes}
    edges = []
    for a_index in range(len(nodes)):
        for b_index in range(a_index + 1, len(nodes)):
            a, b = nodes[a_index], nodes[b_index]
            p = p_in if truth[a] == truth[b] else p_out
            if rng.random() < p:
                edges.append((a, b, 1.0))
    return edges, truth


# ---- small graphs with known answers ----

def test_two_triangles_split_at_the_bridge():
    result = detect_communities(TWO_TRIANGLES)
    assert groups(result) == [["a1", "a2", "a3"], ["b1", "b2", "b3"]]
    assert [c["size"] for c in result] == [3, 3]
    assert [c["internal_weight"] for c in result] == [3.0, 3.0]
    assert modularity(TWO_TRIANGLES, groups(result)) == pytest.approx(0.357142857142857)


def test_weighted_pairs_follow_the_heavy_edges():
    assert groups(detect_communities(WEIGHTED_PAIRS)) == [["a", "b"], ["c", "d"]]


def test_a_tie_goes_to_the_smaller_label_and_larger_communities_come_first():
    assert groups(detect_communities(TIE)) == [["a", "b", "m"], ["c", "d"]]


def test_resolution_changes_the_result():
    assert groups(detect_communities(TWO_TRIANGLES, resolution=0.1)) == [["a1", "a2", "a3", "b1", "b2", "b3"]]
    assert len(detect_communities(TWO_TRIANGLES, resolution=1.0)) == 2


@pytest.mark.parametrize("edges", [TWO_TRIANGLES, WEIGHTED_PAIRS, TIE])
def test_small_graphs_reach_the_best_possible_modularity(edges):
    nodes = sorted({n for a, b, _w in edges for n in (a, b)})
    best = max(modularity(edges, p) for p in all_partitions(nodes))
    ours = modularity(edges, groups(detect_communities(edges, min_size=1)))
    assert ours == pytest.approx(best, abs=1e-9)


def test_ring_of_cliques_gives_one_community_per_clique():
    edges = []
    for k in range(5):
        nodes = [f"c{k}n{i}" for i in range(4)]
        edges += [(a, b, 1.0) for i, a in enumerate(nodes) for b in nodes[i + 1:]]
        edges.append((f"c{k}n0", f"c{(k + 1) % 5}n1", 1.0))
    assert sorted(groups(detect_communities(edges))) == [[f"c{k}n{i}" for i in range(4)] for k in range(5)]


def test_planted_partition_is_recovered():
    edges, truth = planted_graph()
    result = detect_communities(edges, min_size=1)
    found = {m: index for index, c in enumerate(result) for m in c["members"]}
    truth = {n: truth[n] for n in found}
    assert len(truth) >= 38
    assert adjusted_rand_index(truth, found) >= 0.9


# ---- input handling ----

def test_empty_input_gives_no_communities():
    assert detect_communities([]) == []


def test_invalid_edges_are_ignored():
    edges = [
        ("a", "b", 1), ("b", "a", 2), ("a", "a", 5),
        ("a", "c", float("nan")), ("a", "c", float("inf")), ("a", "c", -1),
        ("a", "c", 0), ("a", "c", True), ("a", "c", "3"),
    ]
    result = detect_communities(edges, min_size=1)
    assert groups(result) == [["a", "b"]]
    assert result[0]["internal_weight"] == 3.0


def test_min_size_filters_small_communities():
    assert len(detect_communities(WEIGHTED_PAIRS, min_size=2)) == 2
    assert detect_communities(WEIGHTED_PAIRS, min_size=3) == []


def test_duplicate_edges_are_merged_and_summed_in_sorted_order():
    edges = [("a", "b", 0.1), ("b", "a", 0.2), ("a", "b", 0.3)]
    expected = {("a", "b"): 0.1 + 0.2 + 0.3}
    for permutation in itertools.permutations(edges):
        assert cm._prepare_edges(permutation) == expected
    assert cm._prepare_edges([("b", "a", 1), ("a", "b", 2)]) == {("a", "b"): 3.0}


@pytest.mark.parametrize("bad", [0, -1, 0.0, float("nan"), float("inf"), True, "1"])
def test_resolution_must_be_a_positive_finite_number(bad):
    with pytest.raises(ValueError):
        detect_communities(TWO_TRIANGLES, resolution=bad)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2"])
def test_min_size_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError):
        detect_communities(TWO_TRIANGLES, min_size=bad)


# ---- determinism and output shape ----

def test_the_result_does_not_depend_on_edge_order_or_direction():
    edges, _ = planted_graph()
    edges = edges + [("g0n00", "g0n01", 0.1), ("g0n01", "g0n00", 0.2), ("g0n00", "g0n01", 0.3)]
    baseline = detect_communities(edges, min_size=1)
    for seed in range(10):
        rng = random.Random(seed)
        shuffled = [(b, a, w) if rng.random() < 0.5 else (a, b, w) for a, b, w in edges]
        rng.shuffle(shuffled)
        assert detect_communities(shuffled, min_size=1) == baseline


def test_every_node_appears_once_and_the_fields_are_consistent():
    edges, _ = planted_graph()
    result = detect_communities(edges, min_size=1)
    nodes = {n for a, b, _w in edges for n in (a, b)}
    seen = [m for c in result for m in c["members"]]
    assert sorted(seen) == sorted(nodes) and len(seen) == len(set(seen))
    for c in result:
        assert c["members"] == sorted(c["members"])
        assert c["size"] == len(c["members"])
        assert c["members_hash"] == hashlib.sha256("\x00".join(c["members"]).encode("utf-8")).hexdigest()
        inside = sum(w for a, b, w in edges if a in c["members"] and b in c["members"])
        assert c["internal_weight"] == pytest.approx(inside)
    order = [(-c["size"], c["members"][0]) for c in result]
    assert order == sorted(order)


def test_members_are_sorted_by_code_point_and_hashed_with_nul_separators():
    result = detect_communities([("\U0001F600", "\uff5e", 1)])
    assert groups(result) == [["\uff5e", "\U0001F600"]]
    assert result[0]["members_hash"] == hashlib.sha256("\uff5e\x00\U0001F600".encode("utf-8")).hexdigest()
    assert members_hash(["x", "y"]) == hashlib.sha256(b"x\x00y").hexdigest()


# ---- modularity ----

def test_modularity_known_values():
    two = [["a1", "a2", "a3"], ["b1", "b2", "b3"]]
    assert modularity(TWO_TRIANGLES, two) == pytest.approx(6 / 7 - 0.5)
    assert modularity(TWO_TRIANGLES, two, resolution=2.0) == pytest.approx(6 / 7 - 1.0)
    assert modularity(TWO_TRIANGLES, [["a1", "a2", "a3", "b1", "b2", "b3"]]) == pytest.approx(0.0)
    singles = [[n] for n in ["a1", "a2", "a3", "b1", "b2", "b3"]]
    assert modularity(TWO_TRIANGLES, singles) == pytest.approx(-34 / 196)
    assert modularity([], []) == 0.0
    with pytest.raises(ValueError):
        modularity(TWO_TRIANGLES, [["a1", "a2", "a3"]])


# ---- the building blocks, tested directly ----

def test_aggregate_folds_internal_edges_into_loops_and_keeps_cross_edges():
    adj = [{1: 2.0, 2: 1.0}, {0: 2.0}, {0: 1.0}]
    new_adj, new_loops, mapping = cm._aggregate(adj, [0.5, 0.0, 0.0], [5, 5, 9])
    assert mapping == [0, 0, 1]
    assert new_loops == [2.5, 0.0]
    assert new_adj == [{1: 1.0}, {0: 1.0}]


@pytest.mark.parametrize("cross, merges", [(4.5, False), (7.0, True)])
def test_loops_count_twice_in_the_degree_when_deciding_to_merge(cross, merges):
    community, moved = cm._local_moving([{1: cross}, {0: cross}], [3.0, 3.0], 1.0)
    assert moved is merges
    assert (community == [1, 1]) is merges


def test_a_tie_never_makes_a_node_oscillate_between_communities(monkeypatch):
    adj = [{1: 5.0, 4: 1.0}, {0: 5.0}, {3: 5.0, 4: 1.0}, {2: 5.0}, {0: 1.0, 2: 1.0}]
    monkeypatch.setattr(cm, "MAX_PASSES", 5)
    assert cm._local_moving(adj, [0.0] * 5, 1.0) == ([1, 1, 3, 3, 1], True)


def test_components_split_a_community_that_is_not_connected():
    assert cm._components([[1], [0], [3], [2]], [7, 7, 7, 7]) == [[0, 1], [2, 3]]
    assert cm._components([[1, 2], [0, 3], [0, 3], [1, 2]], [0, 1, 0, 1]) == [[0, 2], [1, 3]]


# ---- comparison with networkx (only where it is installed) ----

def _nx_graph(seed):
    nx = pytest.importorskip("networkx")
    rng = random.Random(seed)
    graph = nx.gnp_random_graph(40, 0.1, seed=seed)
    for u, v in graph.edges():
        graph[u][v]["weight"] = round(rng.uniform(0.5, 3.0), 3)
    return nx, nx.relabel_nodes(graph, {i: f"n{i:02d}" for i in graph.nodes})


@pytest.mark.parametrize("seed", range(6))
def test_modularity_agrees_with_networkx_and_matches_its_louvain(seed):
    nx, graph = _nx_graph(seed)
    edges = [(a, b, d["weight"]) for a, b, d in graph.edges(data=True)]
    parts = groups(detect_communities(edges, min_size=1))
    covered = {m for p in parts for m in p}
    partition = [set(p) for p in parts] + [{n} for n in graph.nodes if n not in covered]
    reference_q = nx.community.modularity(graph, partition, weight="weight")
    assert modularity(edges, parts) == pytest.approx(reference_q, abs=1e-9)
    nx_parts = nx.community.louvain_communities(graph, weight="weight", seed=seed)
    assert reference_q >= nx.community.modularity(graph, nx_parts, weight="weight") - 0.03


def test_karate_club_modularity():
    nx = pytest.importorskip("networkx")
    edges = [(f"k{u:02d}", f"k{v:02d}", 1.0) for u, v in nx.karate_club_graph().edges()]
    assert modularity(edges, groups(detect_communities(edges, min_size=1))) >= 0.40
