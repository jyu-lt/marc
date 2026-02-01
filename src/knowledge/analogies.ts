import { readJsonFile } from "../utils/json.js";
import { cosineSimilarity } from "../utils/similarity.js";
import type { LLMClient } from "../llm/client.js";

export type Analogy = {
  id: string;
  source_domain: string;
  target_domain?: string;
  events: Array<{ event: string; maps_to: string }>;
  lessons: string[];
  conditions_for_applicability: string[];
  embedding?: number[];
};

export type AnalogyIndexData = {
  analogies: Analogy[];
};

export class AnalogyIndex {
  private data: AnalogyIndexData;

  constructor(data: AnalogyIndexData) {
    this.data = data;
  }

  static async load(filePath: string): Promise<AnalogyIndex> {
    const data = await readJsonFile<AnalogyIndexData>(filePath, {
      analogies: []
    });
    return new AnalogyIndex(data);
  }

  get analogies(): Analogy[] {
    return this.data.analogies;
  }

  async matchAnalogies(
    query: string,
    llm?: LLMClient,
    topK = 3
  ): Promise<Array<{ analogy: Analogy; score: number }>> {
    if (this.data.analogies.length === 0) {
      return [];
    }
    if (!llm) {
      return this.keywordMatch(query, topK);
    }

    await this.ensureEmbeddings(llm);
    const queryEmbedding = (await llm.embed(query)) as number[];
    return this.data.analogies
      .map((analogy) => ({
        analogy,
        score: cosineSimilarity(queryEmbedding, analogy.embedding ?? [])
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  private async ensureEmbeddings(llm: LLMClient): Promise<void> {
    const toEmbed = this.data.analogies.filter(
      (analogy) => !analogy.embedding || analogy.embedding.length === 0
    );
    if (toEmbed.length === 0) {
      return;
    }

    const inputs = toEmbed.map((analogy) => {
      return [
        analogy.source_domain,
        analogy.target_domain ?? "",
        analogy.conditions_for_applicability.join(" "),
        analogy.lessons.join(" ")
      ]
        .filter(Boolean)
        .join(" ");
    });

    const embeddings = (await llm.embed(inputs)) as number[][];
    toEmbed.forEach((analogy, index) => {
      analogy.embedding = embeddings[index] ?? [];
    });
  }

  private keywordMatch(
    query: string,
    topK: number
  ): Array<{ analogy: Analogy; score: number }> {
    const needle = query.toLowerCase();
    return this.data.analogies
      .map((analogy) => {
        const haystack =
          `${analogy.source_domain} ${analogy.target_domain ?? ""} ${analogy.lessons.join(" ")} ${analogy.conditions_for_applicability.join(" ")}`.toLowerCase();
        const score = scoreMatch(needle, haystack);
        return { analogy, score };
      })
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }
}

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
