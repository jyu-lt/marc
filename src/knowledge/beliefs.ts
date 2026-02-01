import { readJsonFile, writeJsonFile } from "../utils/json.js";
import { cosineSimilarity } from "../utils/similarity.js";
import { newId } from "../utils/ids.js";
import type { LLMClient } from "../llm/client.js";

export type RelationshipType =
  | "SUPPORTS"
  | "CONTRADICTS"
  | "DEPENDS_ON"
  | "EVOLVED_TO"
  | "ANALOGOUS_TO";

export type Relationship = {
  from: string;
  to: string;
  type: RelationshipType;
  strength: number;
  bidirectional: boolean;
};

export type BeliefNode = {
  id: string;
  statement: string;
  domain_tags: string[];
  confidence: number;
  first_stated: string;
  last_updated: string;
  source_refs: string[];
  embedding?: number[];
};

export type BeliefGraphData = {
  nodes: BeliefNode[];
  relationships: Relationship[];
};

export class BeliefGraph {
  private data: BeliefGraphData;
  private filePath: string;

  constructor(data: BeliefGraphData, filePath: string) {
    this.data = data;
    this.filePath = filePath;
  }

  static async load(filePath: string): Promise<BeliefGraph> {
    const data = await readJsonFile<BeliefGraphData>(filePath, {
      nodes: [],
      relationships: []
    });
    return new BeliefGraph(data, filePath);
  }

  async save(): Promise<void> {
    await writeJsonFile(this.filePath, this.data);
  }

  get nodes(): BeliefNode[] {
    return this.data.nodes;
  }

  get relationships(): Relationship[] {
    return this.data.relationships;
  }

  async addBelief(
    belief: Omit<BeliefNode, "id" | "first_stated" | "last_updated"> & {
      id?: string;
      first_stated?: string;
      last_updated?: string;
    },
    llm?: LLMClient
  ): Promise<BeliefNode> {
    const existing = this.findByStatement(belief.statement);
    if (existing) {
      existing.last_updated = new Date().toISOString();
      existing.confidence = belief.confidence;
      existing.source_refs = Array.from(
        new Set([...existing.source_refs, ...belief.source_refs])
      );
      if (belief.domain_tags.length) {
        existing.domain_tags = Array.from(
          new Set([...existing.domain_tags, ...belief.domain_tags])
        );
      }
      return existing;
    }

    const now = new Date().toISOString();
    const node: BeliefNode = {
      id: belief.id ?? newId(),
      statement: belief.statement,
      domain_tags: belief.domain_tags,
      confidence: belief.confidence,
      first_stated: belief.first_stated ?? now,
      last_updated: belief.last_updated ?? now,
      source_refs: belief.source_refs
    };

    if (llm) {
      const embedding = await llm.embed(node.statement);
      if (Array.isArray(embedding)) {
        node.embedding = embedding as number[];
      }
    }

    this.data.nodes.push(node);
    return node;
  }

  addRelationship(relationship: Relationship): void {
    this.data.relationships.push(relationship);
  }

  async getRelatedBeliefs(
    query: string,
    llm?: LLMClient,
    topK = 5
  ): Promise<BeliefNode[]> {
    if (!llm) {
      return this.keywordMatch(query, topK);
    }

    const queryEmbedding = (await llm.embed(query)) as number[];
    const scored = this.data.nodes
      .filter((node) => node.embedding && node.embedding.length)
      .map((node) => ({
        node,
        score: cosineSimilarity(queryEmbedding, node.embedding as number[])
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((item) => item.node);

    if (scored.length > 0) {
      return scored;
    }

    return this.keywordMatch(query, topK);
  }

  checkConsistency(statement: string): BeliefNode[] {
    const normalized = normalize(statement);
    const negated = normalized.startsWith("not ") ? normalized.slice(4) : `not ${normalized}`;
    return this.data.nodes.filter((node) => {
      const nodeNorm = normalize(node.statement);
      return nodeNorm === negated;
    });
  }

  getTemporalEvolution(tag: string): BeliefNode[] {
    return this.data.nodes
      .filter((node) => node.domain_tags.includes(tag))
      .sort((a, b) => a.first_stated.localeCompare(b.first_stated));
  }

  traceReasoningPath(): BeliefNode[] {
    return [];
  }

  private keywordMatch(query: string, topK: number): BeliefNode[] {
    const needle = normalize(query);
    const scored = this.data.nodes
      .map((node) => ({
        node,
        score: keywordScore(needle, normalize(node.statement), node.domain_tags)
      }))
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((item) => item.node);

    return scored;
  }

  private findByStatement(statement: string): BeliefNode | undefined {
    const normalized = normalize(statement);
    return this.data.nodes.find((node) => normalize(node.statement) === normalized);
  }
}

function normalize(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, "").trim();
}

function keywordScore(query: string, statement: string, tags: string[]): number {
  let score = 0;
  for (const token of query.split(/\s+/)) {
    if (token.length < 3) {
      continue;
    }
    if (statement.includes(token)) {
      score += 2;
    }
    if (tags.some((tag) => tag.toLowerCase().includes(token))) {
      score += 1;
    }
  }
  return score;
}
