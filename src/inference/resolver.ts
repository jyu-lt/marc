import type { BeliefGraph, BeliefNode } from "../knowledge/beliefs.js";
import type { LLMClient } from "../llm/client.js";

export type ResolvedBeliefs = {
  beliefs: BeliefNode[];
  conflicts: BeliefNode[];
};

export async function resolveBeliefs(
  query: string,
  graph: BeliefGraph,
  llm?: LLMClient,
  topK = 5
): Promise<ResolvedBeliefs> {
  // 1. Retrieve related beliefs
  const retrieved = await graph.getRelatedBeliefs(query, llm, topK);

  // 2. Identify conflicts and supports between beliefs
  const resolved: BeliefNode[] = [];
  const conflicts: BeliefNode[] = [];

  // Sort by confidence desc, then by date desc (prefer most stable/recent)
  const candidates = [...retrieved].sort((a, b) => {
    if (b.confidence !== a.confidence) return b.confidence - a.confidence;
    return b.last_updated.localeCompare(a.last_updated);
  });

  for (const node of candidates) {
    const consistency = await graph.checkConsistency(node.statement, llm);

    // Check if this node contradicts any already accepted beliefs
    const isContradicted = resolved.some((r) =>
      consistency.contradictions.some((c) => c.id === r.id)
    );

    if (isContradicted) {
      // Temporal evolution: if existing is older and lower confidence, replace it?
      // For now, simpler: keep highest confidence, prefer newer if same confidence.
      conflicts.push(node);
    } else {
      resolved.push(node);
    }
  }

  return {
    beliefs: resolved,
    conflicts: conflicts
  };
}
