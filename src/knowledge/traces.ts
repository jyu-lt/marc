import { readJsonFile, writeJsonFile } from "../utils/json.js";
import { cosineSimilarity } from "../utils/similarity.js";
import { newId } from "../utils/ids.js";
import { VectorStore } from "./vector_store.js";
import type { LLMClient } from "../llm/client.js";

export type ReasoningTrace = {
  id: string;
  input_context: string;
  framework_selection: {
    chosen: string[];
    selection_rationale?: string;
  };
  analogy_used?: {
    source_domain: string;
    mapping?: Record<string, string>;
  };
  synthesis_steps: string[];
  conclusion: string;
  confidence: number;
  beliefs_invoked: string[];
  source_refs: string[];
  extracted_at: string;
  embedding?: number[] | Float32Array;
};

export type ReasoningTraceStoreData = {
  traces: ReasoningTrace[];
};

export class ReasoningTraceStore {
  private data: ReasoningTraceStoreData;
  private filePath: string;
  private vectorStore: VectorStore;

  constructor(data: ReasoningTraceStoreData, filePath: string) {
    this.data = data;
    this.filePath = filePath;
    this.vectorStore = new VectorStore();
  }

  static async load(filePath: string): Promise<ReasoningTraceStore> {
    const data = await readJsonFile<ReasoningTraceStoreData>(filePath, {
      traces: []
    });
    const store = new ReasoningTraceStore(data, filePath);

    const binaryPath = filePath.replace(/\.json$/, ".bin");
    const vectors = await store.vectorStore.load(binaryPath);

    if (vectors.size > 0) {
      for (const trace of store.data.traces) {
        const vec = vectors.get(trace.id);
        if (vec) {
          trace.embedding = vec;
        }
      }
    } else {
      // Migration: Convert existing number[] to Float32Array
      for (const trace of store.data.traces) {
        if (trace.embedding && Array.isArray(trace.embedding)) {
          trace.embedding = new Float32Array(trace.embedding);
        }
      }
    }

    return store;
  }

  async save(): Promise<void> {
    const binaryPath = this.filePath.replace(/\.json$/, ".bin");
    const embeddingsMap = new Map<string, Float32Array>();

    for (const trace of this.data.traces) {
      if (trace.embedding) {
        const vector = trace.embedding instanceof Float32Array
          ? trace.embedding
          : new Float32Array(trace.embedding);
        embeddingsMap.set(trace.id, vector);
        delete trace.embedding;
      }
    }

    try {
      await writeJsonFile(this.filePath, this.data);
      if (embeddingsMap.size > 0) {
        await this.vectorStore.save(binaryPath, embeddingsMap);
      }
    } finally {
      for (const trace of this.data.traces) {
        const vector = embeddingsMap.get(trace.id);
        if (vector) {
          trace.embedding = vector;
        }
      }
    }
  }

  get traces(): ReasoningTrace[] {
    return this.data.traces;
  }

  async addTrace(
    trace: Omit<ReasoningTrace, "id" | "extracted_at"> & {
      id?: string;
      extracted_at?: string;
    },
    llm?: LLMClient
  ): Promise<ReasoningTrace> {
    const normalizedKey = normalize(`${trace.input_context}||${trace.conclusion ?? ""}`);
    const existingByText = this.data.traces.find(
      (item) => normalize(`${item.input_context}||${item.conclusion ?? ""}`) === normalizedKey
    );
    if (existingByText) {
      mergeTrace(existingByText, trace);
      return existingByText;
    }

    let embedding: Float32Array | undefined;
    if (llm) {
      const vector = await llm.embed(traceTextForEmbedding(trace)) as number[];
      if (vector && vector.length > 0) {
        embedding = new Float32Array(vector);
      }
    }

    if (embedding) {
      const candidate = this.data.traces
        .filter((item) => item.embedding && item.embedding.length === embedding!.length)
        .map((item) => ({
          trace: item,
          score: cosineSimilarity(embedding!, item.embedding!)
        }))
        .sort((a, b) => b.score - a.score)[0];

      if (candidate && candidate.score >= 0.96) {
        mergeTrace(candidate.trace, trace, embedding);
        return candidate.trace;
      }
    }

    const now = new Date().toISOString();
    const stored: ReasoningTrace = {
      id: trace.id ?? newId(),
      input_context: trace.input_context,
      framework_selection: trace.framework_selection ?? { chosen: [] },
      analogy_used: trace.analogy_used,
      synthesis_steps: trace.synthesis_steps ?? [],
      conclusion: trace.conclusion ?? "",
      confidence: Number.isFinite(trace.confidence) ? trace.confidence : 0.5,
      beliefs_invoked: trace.beliefs_invoked ?? [],
      source_refs: trace.source_refs ?? [],
      extracted_at: trace.extracted_at ?? now,
      embedding
    };

    this.data.traces.push(stored);
    return stored;
  }

  async findSimilarTraces(
    query: string,
    llm?: LLMClient,
    topK = 5
  ): Promise<ReasoningTrace[]> {
    if (!llm) {
      return this.keywordMatch(query, topK);
    }

    const queryEmbedding = (await llm.embed(query)) as number[];
    const scored = this.data.traces
      .filter((trace) => trace.embedding && trace.embedding.length)
      .map((trace) => ({
        trace,
        score: cosineSimilarity(queryEmbedding, trace.embedding!)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((item) => item.trace);

    if (scored.length > 0) {
      return scored;
    }

    return this.keywordMatch(query, topK);
  }

  getByFramework(id: string): ReasoningTrace[] {
    return this.data.traces.filter((trace) =>
      trace.framework_selection?.chosen?.includes(id)
    );
  }

  getByBelief(id: string): ReasoningTrace[] {
    return this.data.traces.filter((trace) =>
      trace.beliefs_invoked?.includes(id)
    );
  }

  private keywordMatch(query: string, topK: number): ReasoningTrace[] {
    const needle = normalize(query);
    const scored = this.data.traces
      .map((trace) => ({
        trace,
        score: keywordScore(needle, trace)
      }))
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map((item) => item.trace);

    return scored;
  }
}

function mergeTrace(
  target: ReasoningTrace,
  incoming: Omit<ReasoningTrace, "id" | "extracted_at"> & { extracted_at?: string },
  embedding?: Float32Array
): void {
  const mergedFrameworks = new Set<string>([
    ...(target.framework_selection?.chosen ?? []),
    ...(incoming.framework_selection?.chosen ?? []),
  ]);
  target.framework_selection = {
    chosen: Array.from(mergedFrameworks),
    selection_rationale:
      target.framework_selection?.selection_rationale ??
      incoming.framework_selection?.selection_rationale
  };

  if (!target.analogy_used && incoming.analogy_used) {
    target.analogy_used = incoming.analogy_used;
  }

  if (incoming.synthesis_steps?.length > (target.synthesis_steps?.length ?? 0)) {
    target.synthesis_steps = incoming.synthesis_steps ?? [];
  }

  if (!target.conclusion && incoming.conclusion) {
    target.conclusion = incoming.conclusion;
  }

  if (Number.isFinite(incoming.confidence)) {
    target.confidence = Math.max(target.confidence, incoming.confidence);
  }

  target.beliefs_invoked = Array.from(
    new Set([...(target.beliefs_invoked ?? []), ...(incoming.beliefs_invoked ?? [])])
  );

  target.source_refs = Array.from(
    new Set([...(target.source_refs ?? []), ...(incoming.source_refs ?? [])])
  );

  target.extracted_at = incoming.extracted_at ?? new Date().toISOString();

  if (!target.embedding && embedding) {
    target.embedding = embedding;
  }
}

function traceTextForEmbedding(trace: {
  input_context: string;
  synthesis_steps?: string[];
  conclusion?: string;
}): string {
  return [
    trace.input_context,
    ...(trace.synthesis_steps ?? []),
    trace.conclusion ?? ""
  ]
    .filter(Boolean)
    .join(" ");
}

function normalize(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9\s]/g, "").trim();
}

function keywordScore(query: string, trace: ReasoningTrace): number {
  const haystack = normalize(
    [
      trace.input_context,
      trace.conclusion,
      ...(trace.synthesis_steps ?? []),
      ...(trace.framework_selection?.chosen ?? []),
      ...(trace.beliefs_invoked ?? [])
    ]
      .filter(Boolean)
      .join(" ")
  );
  let score = 0;
  for (const token of query.split(/\s+/)) {
    if (token.length < 3) {
      continue;
    }
    if (haystack.includes(token)) {
      score += 1;
    }
  }
  return score;
}
