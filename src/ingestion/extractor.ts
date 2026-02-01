import { z } from "zod";
import type { Segment } from "./segmenter.js";
import type { LLMClient } from "../llm/client.js";

type UnknownRecord = Record<string, unknown>;

function asRecord(value: unknown): UnknownRecord | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as UnknownRecord;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function toStringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.filter((item): item is string => typeof item === "string");
  }
  if (typeof value === "string") {
    return [value];
  }
  return [];
}

function clamp01(value: number): number {
  return Math.min(Math.max(value, 0), 1);
}

function coerceConfidence(value: unknown): number | string | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return clamp01(value);
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) {
      return clamp01(numeric);
    }
    return trimmed;
  }
  return undefined;
}

function coerceConfidenceNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return clamp01(value);
  }
  if (typeof value === "string") {
    const trimmed = value.trim().toLowerCase();
    const numeric = Number(trimmed);
    if (Number.isFinite(numeric)) {
      return clamp01(numeric);
    }
    if (trimmed.includes("high")) {
      return 0.8;
    }
    if (trimmed.includes("medium")) {
      return 0.6;
    }
    if (trimmed.includes("low")) {
      return 0.4;
    }
  }
  return undefined;
}

function normalizeFrameworks(value: unknown): UnknownRecord[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const normalized: UnknownRecord[] = [];
  for (const entry of value) {
    if (typeof entry === "string") {
      normalized.push({
        name: entry,
        application: "",
        trigger_conditions: "",
      });
      continue;
    }
    const record = asRecord(entry);
    if (!record) {
      continue;
    }
    const name = asString(record.name);
    if (!name) {
      continue;
    }
    normalized.push({
      name,
      application: asString(record.application) ?? "",
      trigger_conditions: asString(record.trigger_conditions) ?? "",
    });
  }
  return normalized;
}

function normalizeHeuristics(value: unknown): UnknownRecord[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const normalized: UnknownRecord[] = [];
  for (const entry of value) {
    if (typeof entry === "string") {
      normalized.push({ pattern: entry, confidence: 0.5 });
      continue;
    }
    const record = asRecord(entry);
    if (!record) {
      continue;
    }
    const pattern = asString(record.pattern) ?? asString(record.heuristic);
    if (!pattern) {
      continue;
    }
    normalized.push({
      pattern,
      confidence: coerceConfidenceNumber(record.confidence) ?? 0.5,
    });
  }
  return normalized;
}

function normalizeBeliefs(value: unknown): UnknownRecord[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const normalized: UnknownRecord[] = [];
  for (const entry of value) {
    if (typeof entry === "string") {
      normalized.push({ belief: entry });
      continue;
    }
    const record = asRecord(entry);
    if (!record) {
      continue;
    }
    const belief =
      asString(record.belief) ??
      asString(record.statement) ??
      asString(record.text);
    if (!belief) {
      continue;
    }
    normalized.push({
      belief,
      confidence: coerceConfidence(record.confidence ?? record.strength),
      stated_date:
        asString(record.stated_date) ??
        asString(record.date) ??
        asString(record.first_stated),
      dependencies: toStringArray(record.dependencies ?? record.depends_on),
      contradicts: toStringArray(record.contradicts ?? record.conflicts),
      evidence_cited: toStringArray(record.evidence_cited ?? record.evidence),
      domain: toStringArray(
        record.domain ?? record.domain_tags ?? record.tags ?? record.category
      ),
    });
  }
  return normalized;
}

function normalizeStringRecord(value: unknown): Record<string, string> | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  const entries = Object.entries(record).filter(
    ([, item]) => typeof item === "string"
  ) as Array<[string, string]>;
  if (!entries.length) {
    return undefined;
  }
  return Object.fromEntries(entries);
}

function normalizeFrameworkSelection(
  value: unknown
): { chosen: string[]; rejected: string[]; selection_rationale?: string } | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  const chosen = toStringArray(record.chosen ?? record.selected ?? record.frameworks);
  const rejected = toStringArray(record.rejected);
  const selection_rationale =
    asString(record.selection_rationale) ?? asString(record.rationale);
  if (!chosen.length && !rejected.length && !selection_rationale) {
    return undefined;
  }
  return { chosen, rejected, selection_rationale };
}

function normalizeAnalogyUsed(
  value: unknown
): { source_domain?: string; mapping?: Record<string, string> } | undefined {
  const record = asRecord(value);
  if (!record) {
    return undefined;
  }
  const source_domain = asString(record.source_domain) ?? asString(record.source);
  const mapping = normalizeStringRecord(record.mapping);
  if (!source_domain && !mapping) {
    return undefined;
  }
  return { source_domain, mapping };
}

function normalizeReasoningTraces(value: unknown): UnknownRecord[] {
  if (!Array.isArray(value)) {
    return [];
  }
  const normalized: UnknownRecord[] = [];
  for (const entry of value) {
    if (typeof entry === "string") {
      normalized.push({ input_context: entry, synthesis_steps: [] });
      continue;
    }
    const record = asRecord(entry);
    if (!record) {
      continue;
    }
    const input_context =
      asString(record.input_context) ??
      asString(record.context) ??
      asString(record.input);
    if (!input_context) {
      continue;
    }
    const framework_selection = normalizeFrameworkSelection(
      record.framework_selection
    );
    const analogy_used = normalizeAnalogyUsed(record.analogy_used);
    normalized.push({
      input_context,
      framework_selection,
      analogy_used,
      synthesis_steps: toStringArray(record.synthesis_steps ?? record.steps),
      conclusion: asString(record.conclusion ?? record.result),
      confidence: coerceConfidenceNumber(record.confidence),
    });
  }
  return normalized;
}

const FrameworkItemSchema = z.object({
  name: z.string(),
  application: z.string(),
  trigger_conditions: z.string(),
});

const HeuristicItemSchema = z.object({
  pattern: z.string(),
  confidence: z.number().min(0).max(1),
});

const BeliefItemSchema = z.object({
  belief: z.string(),
  confidence: z.union([z.string(), z.number()]).optional(),
  stated_date: z.string().optional(),
  dependencies: z.array(z.string()).default([]),
  contradicts: z.array(z.string()).default([]),
  evidence_cited: z.array(z.string()).default([]),
  domain: z.array(z.string()).default([]),
});

const ReasoningTraceSchema = z.object({
  input_context: z.string(),
  framework_selection: z
    .object({
      chosen: z.array(z.string()).default([]),
      rejected: z.array(z.string()).default([]),
      selection_rationale: z.string().optional(),
    })
    .optional(),
  analogy_used: z
    .object({
      source_domain: z.string().optional(),
      mapping: z.record(z.string()).optional(),
    })
    .optional(),
  synthesis_steps: z.array(z.string()).default([]),
  conclusion: z.string().optional(),
  confidence: z.number().min(0).max(1).optional(),
});

export const ExtractionSchema = z.object({
  frameworks_invoked: z.preprocess(
    normalizeFrameworks,
    z.array(FrameworkItemSchema).default([])
  ),
  implicit_heuristics: z.preprocess(
    normalizeHeuristics,
    z.array(HeuristicItemSchema).default([])
  ),
  beliefs: z.preprocess(
    normalizeBeliefs,
    z.array(BeliefItemSchema).default([])
  ),
  reasoning_traces: z.preprocess(
    normalizeReasoningTraces,
    z.array(ReasoningTraceSchema).default([])
  ),
});

export type ExtractionResult = z.infer<typeof ExtractionSchema>;

export async function extractFromSegment(
  segment: Segment,
  llm?: LLMClient
): Promise<ExtractionResult> {
  if (!llm) {
    return ExtractionSchema.parse({});
  }

  const prompt = `Extract reasoning artifacts from the segment below.\n\nSegment:\n${segment.text}\n\nReturn JSON with keys: frameworks_invoked, implicit_heuristics, beliefs, reasoning_traces.`;

  return llm.generateJSON({
    system: "You extract structured beliefs and frameworks from text.",
    user: prompt,
    schema: ExtractionSchema
  });
}
