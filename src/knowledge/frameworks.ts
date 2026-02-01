import { readJsonFile } from "../utils/json.js";
import { averageEmbedding, cosineSimilarity } from "../utils/similarity.js";
import type { LLMClient } from "../llm/client.js";

export type Framework = {
  id: string;
  name: string;
  trigger_conditions: string[];
  reasoning_steps: string[];
  typical_conclusions: string[];
  counter_conditions: string[];
  trigger_embedding?: number[];
};

export type FrameworkLibraryData = {
  frameworks: Framework[];
};

export class FrameworkLibrary {
  private data: FrameworkLibraryData;

  constructor(data: FrameworkLibraryData) {
    this.data = data;
  }

  static async load(filePath: string): Promise<FrameworkLibrary> {
    const data = await readJsonFile<FrameworkLibraryData>(filePath, {
      frameworks: []
    });
    return new FrameworkLibrary(data);
  }

  get frameworks(): Framework[] {
    return this.data.frameworks;
  }

  async matchFrameworks(
    query: string,
    llm?: LLMClient,
    topK = 3
  ): Promise<Array<{ framework: Framework; score: number }>> {
    if (this.data.frameworks.length === 0) {
      return [];
    }

    if (!llm) {
      return this.keywordMatch(query, topK);
    }

    await this.ensureEmbeddings(llm);
    const queryEmbedding = (await llm.embed(query)) as number[];
    const scored = this.data.frameworks
      .map((framework) => ({
        framework,
        score: cosineSimilarity(queryEmbedding, framework.trigger_embedding ?? [])
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    return scored;
  }

  private async ensureEmbeddings(llm: LLMClient): Promise<void> {
    const toEmbed = this.data.frameworks.filter(
      (framework) => !framework.trigger_embedding || framework.trigger_embedding.length === 0
    );
    if (toEmbed.length === 0) {
      return;
    }

    const inputs = toEmbed.map((framework) => framework.trigger_conditions.join(" "));
    const embeddings = (await llm.embed(inputs)) as number[][];
    toEmbed.forEach((framework, index) => {
      const embedding = embeddings[index] ?? [];
      framework.trigger_embedding = averageEmbedding([embedding]);
    });
  }

  private keywordMatch(
    query: string,
    topK: number
  ): Array<{ framework: Framework; score: number }> {
    const needle = query.toLowerCase();
    return this.data.frameworks
      .map((framework) => {
        const haystack =
          framework.trigger_conditions.join(" ").toLowerCase() +
          " " +
          framework.name.toLowerCase();
        const score = scoreMatch(needle, haystack);
        return { framework, score };
      })
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
}

/**
 * Basic keyword-based scoring as a fallback when LLM/embeddings are unavailable.
 * Matches tokens longer than 2 characters against the concatenated text.
 */
function scoreMatch(query: string, text: string): number {
  let score = 0;
  for (const token of query.split(/\s+/)) {
    if (token.length < 3) {
      continue;
    }
    if (text.includes(token)) {
      score += 1;
    }
  }
  return score;
}
