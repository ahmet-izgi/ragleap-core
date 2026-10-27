export interface GraphConfigOptions {
  baseUrl?: string;
  user?: string;
  password?: string;
  database?: string;
  requestTimeoutMs?: number;
}

/**
 * Connection settings for the Neo4j HTTP Query API (not a Bolt address).
 * The password is a private field: it is never printed by toString, JSON.stringify or util.inspect.
 */
export class GraphConfig {
  readonly baseUrl: string;
  readonly user: string;
  readonly database: string;
  readonly requestTimeoutMs: number;
  #password: string;

  constructor(options: GraphConfigOptions = {}) {
    this.baseUrl = options.baseUrl ?? "http://localhost:7474";
    this.user = options.user ?? "neo4j";
    this.#password = options.password ?? "";
    const db = options.database;
    this.database = db === undefined || db.trim() === "" ? "neo4j" : db;
    this.requestTimeoutMs = options.requestTimeoutMs ?? 60_000;
  }

  get password(): string {
    return this.#password;
  }

  toString(): string {
    return `GraphConfig[baseUrl=${this.baseUrl}, user=${this.user}, password=***, database=${this.database}]`;
  }

  toJSON(): Record<string, string> {
    return { baseUrl: this.baseUrl, user: this.user, password: "***", database: this.database };
  }
}
