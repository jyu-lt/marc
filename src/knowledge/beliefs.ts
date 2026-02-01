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

  async checkConsistency(
    statement: string,
    llm?: LLMClient,
    topK = 5
  ): Promise<{ contradictions: BeliefNode[]; supports: BeliefNode[] }> {
    const results: { contradictions: BeliefNode[]; supports: BeliefNode[] } = {
      contradictions: [],
      supports: []
    };

    // 1. Basic lexical negation check (fallback)
    const normalized = normalize(statement);
    const negated = normalized.startsWith("not ") ? normalized.slice(4) : `not ${normalized}`;
    const lexicallMatches = this.data.nodes.filter((node) => {
      const nodeNorm = normalize(node.statement);
      return nodeNorm === negated;
    });
    results.contradictions.push(...lexicallMatches);

    if (!llm) {
      return results;
    }

    // 2. Semantic consistency check using LLM
    const related = await this.getRelatedBeliefs(statement, llm, topK);
    const candidates = related.filter((node) => !results.contradictions.some((c) => c.id === node.id));

    if (candidates.length === 0) {
      return results;
    }

    const prompt = `Classify the relationship between the follow statement and each candidate belief.
Statement: "${statement}"

Candidates:
${candidates.map((c, i) => `${i + 1}. "${c.statement}"`).join("\n")}

For each, respond ONLY with one of: [SUPPORTS, CONTRADICTS, NEUTRAL]
Format: 1. RELATIONSHIP
2. RELATIONSHIP
...`;

    const response = await llm.generateText(
      "You are a consistency checker. Classify the relationship between statements.",
      prompt
    );
    const classifications = response
      .split("\n")
      .map((line: string) => line.split(".")[1]?.trim().toUpperCase())
      .filter(Boolean);

    candidates.forEach((node, i) => {
      const type = classifications[i];
      if (type === "CONTRADICTS") {
        results.contradictions.push(node);
      } else if (type === "SUPPORTS") {
        results.supports.push(node);
      }
    });

    return results;
  }

  getTemporalEvolution(tag: string): BeliefNode[] {
    return this.data.nodes
      .filter((node) => node.domain_tags.includes(tag))
      .sort((a, b) => a.first_stated.localeCompare(b.first_stated));
  }

  traceReasoningPath(beliefId: string): BeliefNode[] {
    const path: BeliefNode[] = [];
    const visited = new Set<string>();
    let currentId: string | undefined = beliefId;

    while (currentId && !visited.has(currentId)) {
      visited.add(currentId);
      const node = this.data.nodes.find((n) => n.id === currentId);
      if (!node) break;

      path.unshift(node); // Origins first

      // Look for what this belief DEPENDS_ON or is SUPPORTED by
      const relationship = this.data.relationships.find(
        (r) => r.to === currentId && (r.type === "DEPENDS_ON" || r.type === "SUPPORTS")
      );

      currentId = relationship?.from;
    }

    return path;
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

/**
 * Basic keyword-based scoring as a fallback when LLM/embeddings are unavailable.
 * Matches tokens longer than 2 characters against statement and tags.
 */
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
