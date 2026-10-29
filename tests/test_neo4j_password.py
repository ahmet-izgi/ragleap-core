"""The Neo4j password: docker-compose.yml takes it from NEO4J_PASSWORD (public default only while unset), and the app
warns when the public default is configured. No Docker and no Neo4j needed."""
import logging
import pathlib

import yaml

from core import graph

ROOT = pathlib.Path(__file__).resolve().parent.parent
PUBLIC = "ragleapgraph"


def test_compose_gives_the_neo4j_container_the_password_from_the_env_file():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert compose["services"]["neo4j"]["environment"]["NEO4J_AUTH"] == "neo4j/${NEO4J_PASSWORD:-%s}" % PUBLIC


def test_the_public_default_is_written_only_as_the_fallback():
    text = (ROOT / "docker-compose.yml").read_text()
    assert text.count(PUBLIC) == 1 and ":-" + PUBLIC + "}" in text


def test_the_default_matches_the_one_the_app_warns_about():
    assert graph.PUBLIC_DEFAULT_NEO4J_PASSWORD == PUBLIC


class _FakeDatabase:
    @staticmethod
    def driver(uri, auth=None):
        return object()


def _make(monkeypatch, password):
    monkeypatch.setattr(graph, "GraphDatabase", _FakeDatabase)
    monkeypatch.setattr(graph, "NEO4J_PASSWORD", password)
    return graph.GraphService()


def test_the_public_default_password_is_warned_about(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=graph.logger.name):
        svc = _make(monkeypatch, PUBLIC)
    assert svc.driver is not None
    assert any("public default" in r.getMessage() for r in caplog.records)


def test_a_chosen_password_gets_no_warning_and_is_never_logged(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger=graph.logger.name):
        svc = _make(monkeypatch, "a-long-random-value-1234567890")
    assert svc.driver is not None
    assert not any("public default" in r.getMessage() for r in caplog.records)
    assert "a-long-random-value" not in caplog.text


def test_no_password_still_means_graph_off(monkeypatch):
    monkeypatch.setattr(graph, "GraphDatabase", _FakeDatabase)
    monkeypatch.setattr(graph, "NEO4J_PASSWORD", "")
    assert graph.GraphService().driver is None


def test_the_example_env_file_and_readme_tell_people_to_choose_a_password():
    assert "openssl rand -hex 24" in (ROOT / ".env.example").read_text()
    assert "openssl rand -hex 24" in (ROOT / "README.md").read_text()
