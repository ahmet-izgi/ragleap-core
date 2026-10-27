import { GraphConfig } from "./config.ts";
import { GraphException, GraphTransientException } from "./errors.ts";
import { encodeNeo4jJson } from "./neo4jJson.ts";

export type Row = Record<string, unknown>;

/** Runs one Cypher statement in its own auto-commit transaction and returns its rows. */
export interface CypherRunner {
  run(cypher: string, params?: Record<string, unknown> | null): Promise<Row[]>;
  close(): void;
}

const MAX_SAFE = BigInt(Number.MAX_SAFE_INTEGER);
const MIN_SAFE = BigInt(Number.MIN_SAFE_INTEGER);
const INTEGER_SOURCE = /^-?\d+$/;

// Integers beyond 2^53 come back as BigInt, everything else as a normal number.
function parseJson(text: string): any {
  return JSON.parse(text, (_key: string, value: unknown, context?: { source?: string }) => {
    const src = context?.source;
    if (typeof value === "number" && src !== undefined && INTEGER_SOURCE.test(src)) {
      const big = BigInt(src);
      if (big > MAX_SAFE || big < MIN_SAFE) return big;
    }
    return value;
  });
}

function basicToken(user: string, password: string): string {
  const bytes = new TextEncoder().encode(`${user}:${password}`);
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin);
}

function describe(e: unknown): string {
  if (e instanceof Error) {
    const cause = (e as { cause?: unknown }).cause;
    return cause instanceof Error ? `${e.message}: ${cause.message}` : e.message;
  }
  return String(e);
}

function parseResponse(status: number, text: string): Row[] {
  let root: any;
  if (text.trim() === "") {
    root = {};
  } else {
    try {
      root = parseJson(text);
    } catch {
      throw new GraphException(`Unexpected response from Neo4j (HTTP ${status})`);
    }
  }
  const errors = root?.errors;
  if (Array.isArray(errors) && errors.length > 0) {
    const code = String(errors[0]?.code ?? "");
    const message = String(errors[0]?.message ?? "");
    const full = code === "" ? message : `${code}: ${message}`;
    if (code.startsWith("Neo.TransientError.")) throw new GraphTransientException(full);
    throw new GraphException(full);
  }
  if (status < 200 || status >= 300) throw new GraphException(`Neo4j returned HTTP ${status}`);
  const fields: unknown[] = Array.isArray(root?.data?.fields) ? root.data.fields : [];
  const valueRows: unknown[] = Array.isArray(root?.data?.values) ? root.data.values : [];
  const rows: Row[] = [];
  for (const values of valueRows) {
    const arr: unknown[] = Array.isArray(values) ? values : [];
    const row: Row = {};
    for (let i = 0; i < fields.length; i++) {
      row[String(fields[i])] = arr[i] === undefined ? null : arr[i];
    }
    rows.push(row);
  }
  return rows;
}

/**
 * CypherRunner over the Neo4j HTTP Query API v2. Every call is its own auto-commit transaction.
 * Single server only: result bookmarks are not passed along.
 */
export class QueryApiRunner implements CypherRunner {
  #endpoint: string;
  #authHeader: string;
  #timeoutMs: number;
  #fetch: typeof fetch;

  constructor(config: GraphConfig, fetchImpl: typeof fetch = fetch) {
    const base = config.baseUrl.replace(/\/+$/, "");
    this.#endpoint = `${base}/db/${encodeURIComponent(config.database)}/query/v2`;
    this.#authHeader = "Basic " + basicToken(config.user, config.password);
    this.#timeoutMs = config.requestTimeoutMs;
    this.#fetch = fetchImpl;
  }

  async run(cypher: string, params: Record<string, unknown> | null = {}): Promise<Row[]> {
    const body = '{"statement":' + JSON.stringify(cypher) + ',"parameters":' + encodeNeo4jJson(params ?? {}) + "}";
    let status: number;
    let text: string;
    try {
      const response = await this.#fetch(this.#endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          Authorization: this.#authHeader,
        },
        body,
        signal: AbortSignal.timeout(this.#timeoutMs),
      });
      status = response.status;
      text = await response.text();
    } catch (e) {
      throw new GraphException("Request to Neo4j failed: " + describe(e), { cause: e });
    }
    return parseResponse(status, text);
  }

  close(): void {}
}
