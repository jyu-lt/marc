import type { FrameworkLibrary, Framework } from "../knowledge/frameworks.js";
import type { LLMClient } from "../llm/client.js";

export type SelectedFramework = {
  framework: Framework;
  score: number;
  rationale: string;
};

export async function selectFrameworks(
  query: string,
  library: FrameworkLibrary,
  llm?: LLMClient,
  topK = 3
): Promise<SelectedFramework[]> {
  const matches = await library.matchFrameworks(query, llm, topK);
  return matches.map((match) => ({
    framework: match.framework,
    score: match.score,
    rationale: match.framework.trigger_conditions.slice(0, 2).join(" | ")
  }));
}
