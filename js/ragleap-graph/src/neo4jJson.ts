/**
 * Marks a number that must reach Neo4j as FLOAT. JavaScript cannot tell 1 from 1.0, so plain
 * integral numbers are sent as INTEGER and anything that must be FLOAT (weights) is wrapped.
 */
export class Neo4jFloat {
  readonly value: number;
  constructor(value: number) {
    this.value = value;
  }
}

export function neo4jFloat(value: number): Neo4jFloat {
  return new Neo4jFloat(value);
}

function assertFinite(v: number): void {
  if (!Number.isFinite(v)) throw new TypeError("Non-finite number in query parameters");
}

function encodeFloat(v: number): string {
  assertFinite(v);
  if (Object.is(v, -0)) return "-0.0";
  const s = String(v);
  return /[.eE]/.test(s) ? s : s + ".0";
}

function encodeNumber(v: number): string {
  assertFinite(v);
  if (Number.isInteger(v) && !Number.isSafeInteger(v)) {
    throw new RangeError("Integer beyond 2^53 must be passed as a BigInt");
  }
  return String(v);
}

function isPlainObject(v: object): boolean {
  const p = Object.getPrototypeOf(v);
  return p === Object.prototype || p === null;
}

/** Encodes query parameters as JSON for the Query API. undefined is sent as null. */
export function encodeNeo4jJson(value: unknown): string {
  if (value === null || value === undefined) return "null";
  switch (typeof value) {
    case "string":
      return JSON.stringify(value);
    case "boolean":
      return value ? "true" : "false";
    case "number":
      return encodeNumber(value);
    case "bigint":
      return value.toString();
    case "object":
      break;
    default:
      throw new TypeError(`Unsupported query parameter type: ${typeof value}`);
  }
  if (value instanceof Neo4jFloat) return encodeFloat(value.value);
  if (Array.isArray(value)) return "[" + value.map((x) => encodeNeo4jJson(x)).join(",") + "]";
  let entries: Array<[unknown, unknown]> | null = null;
  if (value instanceof Map) entries = Array.from(value.entries());
  else if (isPlainObject(value as object)) entries = Object.entries(value as object);
  if (entries === null) throw new TypeError("Unsupported query parameter object");
  return "{" + entries.map(([k, v]) => JSON.stringify(String(k)) + ":" + encodeNeo4jJson(v)).join(",") + "}";
}
