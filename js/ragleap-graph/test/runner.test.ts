import http from "node:http";
import { test } from "node:test";
import assert from "node:assert/strict";
import { inspect } from "node:util";
import { GraphConfig } from "../src/config.ts";
import { GraphException, GraphTransientException } from "../src/errors.ts";
import { neo4jFloat } from "../src/neo4jJson.ts";
import { QueryApiRunner } from "../src/queryApiRunner.ts";

interface Fake {
  url: string;
  status: number;
  responseBody: string;
  delayMs: number;
  lastBody: string;
  lastAuth: string | undefined;
  lastPath: string | undefined;
  lastMethod: string | undefined;
  lastContentType: string | undefined;
  lastAccept: string | undefined;
}

async function withFake(fn: (fake: Fake) => Promise<void>): Promise<void> {
  const fake: Fake = {
    url: "", status: 200, responseBody: "{}", delayMs: 0, lastBody: "",
    lastAuth: undefined, lastPath: undefined, lastMethod: undefined,
    lastContentType: undefined, lastAccept: undefined,
  };
  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      fake.lastBody = Buffer.concat(chunks).toString("utf8");
      fake.lastAuth = req.headers.authorization;
      fake.lastPath = req.url;
      fake.lastMethod = req.method;
      fake.lastContentType = req.headers["content-type"];
      fake.lastAccept = req.headers.accept;
      const send = () => {
        if (res.writableEnded || res.destroyed) return;
        res.writeHead(fake.status, { "Content-Type": "application/json" });
        res.end(fake.responseBody);
      };
      if (fake.delayMs > 0) setTimeout(send, fake.delayMs);
      else send();
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", () => resolve()));
  const port = (server.address() as { port: number }).port;
  fake.url = `http://127.0.0.1:${port}`;
  try {
    await fn(fake);
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
}

const EMPTY = '{"data":{"fields":[],"values":[]}}';

test("maps rows and value types", async () => {
  await withFake(async (fake) => {
    fake.responseBody =
      '{"data":{"fields":["i","f","s","n","l","m"],"values":[[1,2.0,"x",null,[1,"a"],{"k":1}]]}}';
    const rows = await new QueryApiRunner(new GraphConfig({ baseUrl: fake.url, password: "example-password" })).run("RETURN 1", {});
    assert.deepStrictEqual(rows, [{ i: 1, f: 2, s: "x", n: null, l: [1, "a"], m: { k: 1 } }]);
  });
});

test("integers beyond 2^53 come back as BigInt, safe ones as numbers", async () => {
  await withFake(async (fake) => {
    fake.responseBody =
      '{"data":{"fields":["big","neg","safe"],"values":[[9007199254740993,-9007199254740993,9007199254740991]]}}';
    const rows = await new QueryApiRunner(new GraphConfig({ baseUrl: fake.url })).run("RETURN 1");
    assert.deepStrictEqual(rows, [{ big: 9007199254740993n, neg: -9007199254740993n, safe: 9007199254740991 }]);
  });
});

test("request body: floats keep the decimal point, integers do not, order is kept", async () => {
  await withFake(async (fake) => {
    fake.responseBody = EMPTY;
    const params = {
      w: neo4jFloat(1), n: 1, big: 9007199254740993n, s: "h\u00e9llo", f: 0.5, z: neo4jFloat(-0),
      list: [1, neo4jFloat(2), "a", null], obj: { k: neo4jFloat(0.5) }, m: new Map([["a", 1]]),
    };
    await new QueryApiRunner(new GraphConfig({ baseUrl: fake.url })).run("RETURN $w", params);
    assert.equal(
      fake.lastBody,
      '{"statement":"RETURN $w","parameters":{"w":1.0,"n":1,"big":9007199254740993,"s":"h\u00e9llo","f":0.5,"z":-0.0,' +
        '"list":[1,2.0,"a",null],"obj":{"k":0.5},"m":{"a":1}}}',
    );
  });
});

test("null params send an empty parameters object", async () => {
  await withFake(async (fake) => {
    fake.responseBody = EMPTY;
    await new QueryApiRunner(new GraphConfig({ baseUrl: fake.url })).run("RETURN 1", null);
    assert.equal(fake.lastBody, '{"statement":"RETURN 1","parameters":{}}');
  });
});

test("sends Basic auth, JSON headers, POST and the Query API path", async () => {
  await withFake(async (fake) => {
    fake.responseBody = EMPTY;
    await new QueryApiRunner(new GraphConfig({ baseUrl: fake.url, user: "neo4j", password: "example-password" })).run("RETURN 1");
    assert.equal(fake.lastAuth, "Basic " + Buffer.from("neo4j:example-password", "utf8").toString("base64"));
    assert.equal(fake.lastPath, "/db/neo4j/query/v2");
    assert.equal(fake.lastMethod, "POST");
    assert.equal(fake.lastContentType, "application/json");
    assert.equal(fake.lastAccept, "application/json");
  });
});

test("non-ASCII credentials are UTF-8 encoded", async () => {
  await withFake(async (fake) => {
    fake.responseBody = EMPTY;
    await new QueryApiRunner(new GraphConfig({ baseUrl: fake.url, user: "\u00fc", password: "example-p\u00e4ss" })).run("RETURN 1");
    assert.equal(fake.lastAuth, "Basic " + Buffer.from("\u00fc:example-p\u00e4ss", "utf8").toString("base64"));
  });
});

test("trailing slashes are removed and the database name is encoded", async () => {
  await withFake(async (fake) => {
    fake.responseBody = EMPTY;
    await new QueryApiRunner(new GraphConfig({ baseUrl: fake.url + "//", database: "my db" })).run("RETURN 1");
    assert.equal(fake.lastPath, "/db/my%20db/query/v2");
  });
});

test("transient and client errors are mapped", async () => {
  await withFake(async (fake) => {
    const runner = new QueryApiRunner(new GraphConfig({ baseUrl: fake.url }));
    fake.status = 409;
    fake.responseBody = '{"errors":[{"code":"Neo.TransientError.Transaction.DeadlockDetected","message":"deadlock"}]}';
    await assert.rejects(
      () => runner.run("X"),
      (e: unknown) =>
        e instanceof GraphTransientException && e.message === "Neo.TransientError.Transaction.DeadlockDetected: deadlock",
    );
    fake.status = 400;
    fake.responseBody = '{"errors":[{"code":"Neo.ClientError.Statement.SyntaxError","message":"bad"}]}';
    await assert.rejects(
      () => runner.run("X"),
      (e: unknown) =>
        e instanceof GraphException && !(e instanceof GraphTransientException) &&
        e.message === "Neo.ClientError.Statement.SyntaxError: bad",
    );
  });
});

test("HTTP error without a body is reported with its status", async () => {
  await withFake(async (fake) => {
    fake.status = 401;
    fake.responseBody = "";
    await assert.rejects(
      () => new QueryApiRunner(new GraphConfig({ baseUrl: fake.url })).run("X"),
      (e: unknown) => e instanceof GraphException && e.message.includes("401"),
    );
  });
});

test("a non-JSON response is reported with its status", async () => {
  await withFake(async (fake) => {
    fake.status = 502;
    fake.responseBody = "<html>bad gateway</html>";
    await assert.rejects(
      () => new QueryApiRunner(new GraphConfig({ baseUrl: fake.url })).run("X"),
      (e: unknown) => e instanceof GraphException && e.message === "Unexpected response from Neo4j (HTTP 502)",
    );
  });
});

test("a refused connection is a GraphException", async () => {
  const dead = new QueryApiRunner(new GraphConfig({ baseUrl: "http://127.0.0.1:1" }));
  await assert.rejects(
    () => dead.run("RETURN 1"),
    (e: unknown) => e instanceof GraphException && e.message.startsWith("Request to Neo4j failed"),
  );
});

test("the request timeout is enforced", async () => {
  await withFake(async (fake) => {
    fake.responseBody = EMPTY;
    fake.delayMs = 400;
    const runner = new QueryApiRunner(new GraphConfig({ baseUrl: fake.url, requestTimeoutMs: 60 }));
    await assert.rejects(
      () => runner.run("RETURN 1"),
      (e: unknown) => e instanceof GraphException && e.message.startsWith("Request to Neo4j failed"),
    );
  });
});

test("non-finite numbers and unsafe integers are rejected before any request", async () => {
  await withFake(async (fake) => {
    const runner = new QueryApiRunner(new GraphConfig({ baseUrl: fake.url }));
    await assert.rejects(() => runner.run("X", { w: NaN }), TypeError);
    await assert.rejects(() => runner.run("X", { w: Infinity }), TypeError);
    await assert.rejects(() => runner.run("X", { w: neo4jFloat(NaN) }), TypeError);
    await assert.rejects(() => runner.run("X", { list: [1, [NaN]] }), TypeError);
    await assert.rejects(() => runner.run("X", { w: 2 ** 60 }), RangeError);
    assert.equal(fake.lastBody, "");
  });
});

test("the password is not printed by toString, JSON or inspect", () => {
  const cfg = new GraphConfig({ baseUrl: "http://h:7474", user: "u", password: "example-password" });
  assert.ok(!String(cfg).includes("example-password"));
  assert.ok(!JSON.stringify(cfg).includes("example-password"));
  assert.ok(!inspect(cfg).includes("example-password"));
  assert.equal(cfg.password, "example-password");
});

test("config defaults", () => {
  const d = new GraphConfig();
  assert.equal(d.baseUrl, "http://localhost:7474");
  assert.equal(d.user, "neo4j");
  assert.equal(d.password, "");
  assert.equal(d.database, "neo4j");
  assert.equal(new GraphConfig({ database: "  " }).database, "neo4j");
});
