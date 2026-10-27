/** Any failure talking to Neo4j (connection, HTTP, or a Neo4j error). */
export class GraphException extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = "GraphException";
  }
}

/** A Neo4j transient error (Neo.TransientError.*), for example a deadlock. Safe to retry. */
export class GraphTransientException extends GraphException {
  constructor(message: string) {
    super(message);
    this.name = "GraphTransientException";
  }
}
