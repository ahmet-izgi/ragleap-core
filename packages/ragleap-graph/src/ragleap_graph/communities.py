"""
ragleap_graph.communities

Deterministic community detection for the entity co-occurrence graph.

Why: a graph of entities answers questions about one entity well, but "what are
the main themes in this corpus?" needs the graph grouped into communities that can
be summarised. Neo4j Community Edition has no Graph Data Science plugin, so the
grouping runs here, on edges read from the graph.

What it does, and what it does not:
  - Louvain modularity optimisation on an undirected weighted graph, then a split
    of every community into its connected components (plain Louvain can return
    communities that are not connected).
  - Fully deterministic: no random numbers; nodes are visited in sorted (code
    point) order, ties go to the smaller community label, and a node only moves
    for a gain larger than 1e-12. The same edges give the same communities
    whatever the input order, so other languages can reproduce the result from
    recorded fixtures.
  - Not Leiden and not hierarchical: one flat level of communities. Modularity
    has a resolution limit (small communities can be merged in large graphs);
    resolution above 1 favours smaller communities.
  - Pure: reads no database and writes nothing.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Sequence, Tuple

Edge = Tuple[str, str, float]

EPSILON = 1e-12
MAX_PASSES = 1000


def _prepare_edges(edges: Iterable[Edge]) -> Dict[Tuple[str, str], float]:
    """
    Undirected weights keyed by (smaller name, larger name). Self-loops and weights
    that are not finite positive numbers (booleans and strings included) are
    dropped; duplicates are summed in sorted order, so float sums do not depend
    on the order the edges came in.
    """
    cleaned: List[Tuple[str, str, float]] = []
    for a, b, weight in edges:
        if a == b:
            continue
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            continue
        if not math.isfinite(weight) or weight <= 0:
            continue
        low, high = (a, b) if a < b else (b, a)
        cleaned.append((low, high, float(weight)))
    cleaned.sort()
    merged: Dict[Tuple[str, str], float] = {}
    for low, high, weight in cleaned:
        merged[(low, high)] = merged.get((low, high), 0.0) + weight
    return merged


def modularity(edges: Iterable[Edge], communities: Sequence[Sequence[str]], resolution: float = 1.0) -> float:
    """
    Modularity of a partition: sum over communities of (inside weight / total weight)
    minus resolution times (community degree / (2 * total weight)) squared.
    Every node that appears in an edge must belong to a community.
    """
    weights = _prepare_edges(edges)
    total = sum(weights.values())
    if total == 0:
        return 0.0
    label: Dict[str, int] = {}
    for index, members in enumerate(communities):
        for name in members:
            label[name] = index
    degree_sum: Dict[int, float] = {}
    inside = 0.0
    for (a, b), weight in weights.items():
        if a not in label or b not in label:
            raise ValueError("every node of an edge must belong to a community")
        degree_sum[label[a]] = degree_sum.get(label[a], 0.0) + weight
        degree_sum[label[b]] = degree_sum.get(label[b], 0.0) + weight
        if label[a] == label[b]:
            inside += weight
    penalty = sum(d * d for d in degree_sum.values()) / (4.0 * total * total)
    return inside / total - resolution * penalty


def _local_moving(adj: List[Dict[int, float]], loops: List[float], resolution: float) -> Tuple[List[int], bool]:
    """
    One Louvain level: move single nodes between communities while that raises
    modularity. adj holds the symmetric neighbour weights, loops the weight inside
    each (aggregated) node; a loop counts twice in its degree.
    """
    n = len(adj)
    degree = [sum(adj[i][j] for j in sorted(adj[i])) + 2.0 * loops[i] for i in range(n)]
    total = sum(degree) / 2.0
    community = list(range(n))
    community_degree = list(degree)
    moved_any = False
    for _ in range(MAX_PASSES):
        moved_pass = False
        for i in range(n):
            own = community[i]
            links: Dict[int, float] = {}
            for j in sorted(adj[i]):
                label = community[j]
                links[label] = links.get(label, 0.0) + adj[i][j]
            community_degree[own] -= degree[i]
            scale = resolution * degree[i] / (2.0 * total)
            best = own
            best_gain = links.get(own, 0.0) - community_degree[own] * scale
            for label in sorted(links):
                if label == own:
                    continue
                gain = links[label] - community_degree[label] * scale
                if gain > best_gain + EPSILON:
                    best, best_gain = label, gain
            community_degree[best] += degree[i]
            if best != own:
                community[i] = best
                moved_pass = True
        if not moved_pass:
            break
        moved_any = True
    return community, moved_any


def _aggregate(adj: List[Dict[int, float]], loops: List[float], community: List[int]):
    """Collapse each community into one node: internal edges become loops, cross edges are summed."""
    order: Dict[int, int] = {}
    for i in range(len(adj)):
        if community[i] not in order:
            order[community[i]] = len(order)
    size = len(order)
    new_adj: List[Dict[int, float]] = [dict() for _ in range(size)]
    new_loops = [0.0] * size
    for i in range(len(adj)):
        ci = order[community[i]]
        new_loops[ci] += loops[i]
        for j in sorted(adj[i]):
            if j < i:
                continue
            cj = order[community[j]]
            weight = adj[i][j]
            if ci == cj:
                new_loops[ci] += weight
            else:
                new_adj[ci][cj] = new_adj[ci].get(cj, 0.0) + weight
                new_adj[cj][ci] = new_adj[cj].get(ci, 0.0) + weight
    mapping = [order[community[i]] for i in range(len(adj))]
    return new_adj, new_loops, mapping


def _louvain(weights: Dict[Tuple[str, str], float], resolution: float):
    names = sorted({name for pair in weights for name in pair})
    index = {name: i for i, name in enumerate(names)}
    adj: List[Dict[int, float]] = [dict() for _ in names]
    for (a, b), weight in weights.items():
        adj[index[a]][index[b]] = weight
        adj[index[b]][index[a]] = weight
    loops = [0.0] * len(names)
    membership = list(range(len(names)))
    while adj:
        community, moved = _local_moving(adj, loops, resolution)
        if not moved:
            break
        adj, loops, mapping = _aggregate(adj, loops, community)
        membership = [mapping[m] for m in membership]
    return names, membership


def _components(neighbors: List[List[int]], membership: List[int]) -> List[List[int]]:
    """Connected parts of every community (edges between different communities do not connect)."""
    seen = [False] * len(membership)
    out: List[List[int]] = []
    for start in range(len(membership)):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        part: List[int] = []
        while stack:
            node = stack.pop()
            part.append(node)
            for other in neighbors[node]:
                if not seen[other] and membership[other] == membership[start]:
                    seen[other] = True
                    stack.append(other)
        out.append(sorted(part))
    return out


def members_hash(members: Sequence[str]) -> str:
    """SHA-256 (hex) of the members joined with NUL characters; pass them sorted by code point."""
    return hashlib.sha256("\x00".join(members).encode("utf-8")).hexdigest()


def detect_communities(edges: Iterable[Edge], resolution: float = 1.0, min_size: int = 2) -> List[Dict]:
    """
    Group the nodes of a weighted undirected graph into communities.

    edges: (name_a, name_b, weight); invalid edges are ignored, see _prepare_edges.
    resolution: above 1 favours smaller communities, below 1 larger ones.
    min_size: communities with fewer members are left out (nodes without any
    edge never appear).

    Returns a list of {"members": [...] sorted by code point, "size", "internal_weight",
    "members_hash"}, largest first, ties by first member.
    """
    if isinstance(resolution, bool) or not isinstance(resolution, (int, float)) or not math.isfinite(resolution) or resolution <= 0:
        raise ValueError(f"resolution must be a positive finite number, got {resolution!r}")
    if isinstance(min_size, bool) or not isinstance(min_size, int) or min_size < 1:
        raise ValueError(f"min_size must be a positive integer, got {min_size!r}")
    weights = _prepare_edges(edges)
    names, membership = _louvain(weights, float(resolution))
    index = {name: i for i, name in enumerate(names)}
    neighbors: List[List[int]] = [[] for _ in names]
    for a, b in weights:
        neighbors[index[a]].append(index[b])
        neighbors[index[b]].append(index[a])
    for row in neighbors:
        row.sort()
    parts = _components(neighbors, membership)
    label = [0] * len(names)
    for part_number, part in enumerate(parts):
        for node in part:
            label[node] = part_number
    internal = [0.0] * len(parts)
    for (a, b), weight in weights.items():
        if label[index[a]] == label[index[b]]:
            internal[label[index[a]]] += weight
    communities: List[Dict] = []
    for part_number, part in enumerate(parts):
        if len(part) < min_size:
            continue
        members = [names[node] for node in part]
        communities.append({
            "members": members,
            "size": len(members),
            "internal_weight": internal[part_number],
            "members_hash": members_hash(members),
        })
    communities.sort(key=lambda c: (-c["size"], c["members"][0]))
    return communities
