import { test } from "node:test";
import assert from "node:assert/strict";
import { GraphConfig } from "../src/config.ts";
import { GraphException, GraphTransientException } from "../src/errors.ts";
import { neo4jFloat } from "../src/neo4jJson.ts";
import { QueryApiRunner } from "../src/queryApiRunner.ts";

// Runs against a real Neo4j test server with read-only queries. Gated: needs NEO4J_HTTP_URL.
const url = process.env.NEO4J_HTTP_URL;
const skip = url ? false : "NEO4J_HTTP_URL not set";

function config(password?: string): GraphConfig {
  return new GraphConfig({
    baseUrl: url as string,
    user: process.env.NEO4J_USER ?? "neo4j",
    password: password ?? process.env.NEO4J_PASSWORD ?? "",
  });
}

test("live: value types and big integers", { skip }, async () => {
  const runner = new QueryApiRunner(config());
  const rows = await runner.run(
    "RETURN 1 AS i, 2.5 AS f, 'x' AS s, null AS n, [1, 'a'] AS l, 9007199254740993 AS big",
    {},
  );
  assert.deepStrictEqual(rows, [{ i: 1, f: 2.5, s: "x", n: null, l: [1, "a"], big: 9007199254740993n }]);
});

test("live: the float marker reaches Neo4j as FLOAT, plain integers as INTEGER", { skip }, async () => {
  const runner = new QueryApiRunner(config());
  const rows = await runner.run("RETURN toString($a) AS fa, toString($b) AS ib, toString($c) AS cb", {
    a: neo4jFloat(1),
    b: 1,
    c: 9007199254740993n,
  });
  assert.deepStrictEqual(rows, [{ fa: "1.0", ib: "1", cb: "9007199254740993" }]);
});

test("live: unicode round trip", { skip }, async () => {
  const runner = new QueryApiRunner(config());
  const text = "h\u00e9llo \u2603 \ud83d\ude00";
  assert.deepStrictEqual(await runner.run("RETURN $s AS s", { s: text }), [{ s: text }]);
});

test("live: a syntax error is a GraphException, not a transient one", { skip }, async () => {
  const runner = new QueryApiRunner(config());
  await assert.rejects(
    () => runner.run("RETRUN 1"),
    (e: unknown) => e instanceof GraphException && !(e instanceof GraphTransientException) && e.message.includes("SyntaxError"),
  );
});

test("live: a wrong password is a GraphException", { skip }, async () => {
  const runner = new QueryApiRunner(config("example-wrong-password"));
  await assert.rejects(
    () => runner.run("RETURN 1"),
    (e: unknown) => e instanceof GraphException && /Unauthorized|401/.test(e.message),
  );
});
