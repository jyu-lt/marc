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

  const selected = await Promise.all(
    matches.map(async (match) => {
      let rationale = match.framework.trigger_conditions.slice(0, 2).join(" | ");

      if (llm) {
        try {
          const prompt = `Explain in one short sentence why the mental framework "${match.framework.name}" (which triggers on: ${match.framework.trigger_conditions.join(", ")}) is relevant to this query: "${query}"`;
          rationale = await llm.generateText(
            "You are a reasoning assistant explaining framework selection.",
            prompt
          );
        } catch (e) {
          // Fallback to default rationale on error
        }
      }

      return {
        framework: match.framework,
        score: match.score,
        rationale: rationale.trim()
      };
    })
  );

  return selected;
}
