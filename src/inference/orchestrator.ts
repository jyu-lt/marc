import { z } from "zod";
import type { LLMClient } from "../llm/client.js";
import type { FrameworkLibrary } from "../knowledge/frameworks.js";
import type { BeliefGraph } from "../knowledge/beliefs.js";
import type { AnalogyIndex } from "../knowledge/analogies.js";
import { selectFrameworks } from "./selector.js";
import { resolveBeliefs } from "./resolver.js";

export const OrchestratorSchema = z.object({
  answer: z.string(),
  frameworks_used: z.array(z.string()).default([]),
  beliefs_used: z.array(z.string()).default([]),
  analogies_used: z.array(z.string()).default([]),
  reasoning_steps: z.array(z.string()).default([]),
  caveats: z.array(z.string()).default([]),
  confidence: z.number().min(0).max(1),
});

export type OrchestratorOutput = z.infer<typeof OrchestratorSchema>;

export class Orchestrator {
  constructor(
    private llm: LLMClient | undefined,
    private frameworks: FrameworkLibrary,
    private beliefs: BeliefGraph,
    private analogies: AnalogyIndex
  ) {}

  async reason(query: string): Promise<OrchestratorOutput> {
    const selectedFrameworks = await selectFrameworks(
      query,
      this.frameworks,
      this.llm,
      3
    );
    const resolved = await resolveBeliefs(query, this.beliefs, this.llm, 5);
    const matchedAnalogies = await this.analogies.matchAnalogies(
      query,
      this.llm,
      3
    );

    if (!this.llm) {
      return OrchestratorSchema.parse({
        answer:
          "LLM client not configured. Set OPENAI_API_KEY to enable synthesis.",
        frameworks_used: selectedFrameworks.map((item) => item.framework.name),
        beliefs_used: resolved.beliefs.map((belief) => belief.statement),
        analogies_used: matchedAnalogies.map(
          (item) => item.analogy.source_domain
        ),
        reasoning_steps: ["Selection only; synthesis skipped."],
        caveats: ["No LLM configured."],
        confidence: 0.2,
      });
    }

    const frameworksPayload = selectedFrameworks.map((item) => ({
      name: item.framework.name,
      trigger_conditions: item.framework.trigger_conditions,
      reasoning_steps: item.framework.reasoning_steps,
      counter_conditions: item.framework.counter_conditions,
    }));

    const beliefsPayload = resolved.beliefs.map((belief) => ({
      statement: belief.statement,
      confidence: belief.confidence,
      domain_tags: belief.domain_tags,
      last_updated: belief.last_updated,
    }));

    const analogiesPayload = matchedAnalogies.map((item) => ({
      source_domain: item.analogy.source_domain,
      target_domain: item.analogy.target_domain,
      lessons: item.analogy.lessons,
      conditions_for_applicability: item.analogy.conditions_for_applicability,
    }));

    const prompt = `You are reasoning about: ${query}\n\nSTEP 1: FRAMEWORK SELECTION\nChosen frameworks: ${JSON.stringify(
      frameworksPayload
    )}\n\nSTEP 2: BELIEF RETRIEVAL\nRelevant positions: ${JSON.stringify(
      beliefsPayload
    )}\n\nSTEP 3: ANALOGY SEARCH\nHistorical patterns: ${JSON.stringify(
      analogiesPayload
    )}\n\nSTEP 4: SYNTHESIS\nApply selected frameworks, incorporate beliefs, and use analogies if applicable.\n\nSTEP 5: CONFIDENCE & CAVEATS\nProvide caveats and what could change the conclusion.\n\nReturn JSON with these exact keys:
- answer (string): your reasoning answer
- frameworks_used (array of strings): just the "name" field from each framework
- beliefs_used (array of strings): just the "statement" field from each belief
- analogies_used (array of strings): just the "source_domain" field from each analogy
- reasoning_steps (array of strings): step-by-step reasoning
- caveats (array of strings): important caveats
- confidence (number 0-1): confidence score`;

    return this.llm.generateJSON({
      system:
        "You are a reasoning engine that outputs concise, structured answers.",
      user: prompt,
      schema: OrchestratorSchema,
    });
  }
}
