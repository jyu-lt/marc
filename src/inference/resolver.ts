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
  const beliefs = await graph.getRelatedBeliefs(query, llm, topK);
  const conflicts = graph.checkConsistency(query);
  return { beliefs, conflicts };
}
