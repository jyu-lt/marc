import "dotenv/config";
import { Command } from "commander";
import path from "path";
import readline from "readline";
import { z } from "zod";
import { LLMClient } from "./llm/client.js";
import { FrameworkLibrary } from "./knowledge/frameworks.js";
import { BeliefGraph } from "./knowledge/beliefs.js";
import { AnalogyIndex } from "./knowledge/analogies.js";
import { ReasoningTraceStore } from "./knowledge/traces.js";
import { Orchestrator } from "./inference/orchestrator.js";
import { loadCorpus, loadDocument } from "./ingestion/loader.js";
import { segmentDocument } from "./ingestion/segmenter.js";
import { extractFromSegment, extractReasoningTrace } from "./ingestion/extractor.js";

const DEFAULT_FRAMEWORKS = "data/frameworks.json";
const DEFAULT_BELIEFS = "data/beliefs.json";
const DEFAULT_ANALOGIES = "data/analogies.json";
const DEFAULT_TRACES = "data/reasoning_traces.json";
const DEFAULT_CORPUS = "data/corpus";
const BELIEF_DOMAIN_TAXONOMY = [
  "ai",
  "regulation",
  "markets",
  "geopolitics",
  "china",
  "startups",
  "media",
  "culture",
  "education",
  "defense",
  "economics",
  "labor",
  "venture",
  "software",
  "platforms",
  "history",
  "crypto",
  "biotech",
  "policy",
  "tech",
];

const program = new Command();
program
  .name("marc")
  .description("Marc reasoning system - Phase 1")
  .version("0.1.0");

program
  .command("ask")
  .argument("<query>", "question to analyze")
  .option("--frameworks <path>", "framework library JSON", DEFAULT_FRAMEWORKS)
  .option("--beliefs <path>", "belief graph JSON", DEFAULT_BELIEFS)
  .option("--analogies <path>", "analogy index JSON", DEFAULT_ANALOGIES)
  .option("--traces <path>", "reasoning trace store JSON", DEFAULT_TRACES)
  .option("--no-llm", "skip LLM synthesis")
  .action(async (query, options) => {
    const llm = buildLLM(options.llm);
    const [frameworks, beliefs, analogies, traces] = await Promise.all([
      FrameworkLibrary.load(options.frameworks),
      BeliefGraph.load(options.beliefs),
      AnalogyIndex.load(options.analogies),
      ReasoningTraceStore.load(options.traces),
    ]);

    const orchestrator = new Orchestrator(
      llm,
      frameworks,
      beliefs,
      analogies,
      traces
    );
    const result = await orchestrator.reason(query);
    console.log(JSON.stringify(result, null, 2));
  });

program
  .command("extract")
  .argument("[path]", "file or directory to extract", DEFAULT_CORPUS)
  .option("--frameworks <path>", "framework library JSON", DEFAULT_FRAMEWORKS)
  .option("--beliefs <path>", "belief graph JSON", DEFAULT_BELIEFS)
  .option("--traces <path>", "reasoning trace store JSON", DEFAULT_TRACES)
  .option(
    "--method <method>",
    "segmentation method (paragraph|llm|markdown)",
    "paragraph"
  )
  .option("--update-beliefs", "merge extracted beliefs into belief graph")
  .option("--extract-traces", "extract reasoning traces into trace store")
  .option("--no-llm", "skip LLM extraction")
  .action(async (targetPath, options) => {
    const llm = buildLLM(options.llm);
    if (options.extractTraces && !llm) {
      console.error("Trace extraction skipped: LLM disabled.");
    }
    const absolutePath = path.resolve(targetPath);
    const stats = await import("fs").then((mod) =>
      mod.promises.stat(absolutePath)
    );
    const docs = stats.isDirectory()
      ? await loadCorpus(absolutePath)
      : [await loadDocument(absolutePath)];

    console.error(`Found ${docs.length} document(s) at ${absolutePath}`);

    const beliefGraph = await BeliefGraph.load(options.beliefs);
    const frameworkLibrary = options.extractTraces
      ? await FrameworkLibrary.load(options.frameworks)
      : undefined;
    const traceStore = options.extractTraces
      ? await ReasoningTraceStore.load(options.traces)
      : undefined;
    const extractions = [] as Array<Record<string, unknown>>;

    for (let i = 0; i < docs.length; i++) {
      const doc = docs[i];
      console.error(
        `[${i + 1}/${docs.length}] Processing document: ${doc.path}`
      );

      const segments = await segmentDocument(doc, {
        method:
          options.method === "llm"
            ? "llm"
            : options.method === "markdown"
            ? "markdown"
            : "paragraph",
        llm,
      });

      console.error(`  - Segmented into ${segments.length} parts`);

      for (let j = 0; j < segments.length; j++) {
        const segment = segments[j];
        process.stderr.write(
          `  - Extracting segment ${j + 1}/${segments.length}... `
        );
        const extracted = await extractFromSegment(segment, llm);
        process.stderr.write(`found ${extracted.beliefs.length} beliefs\n`);

        let traceResult: Awaited<ReturnType<typeof extractReasoningTrace>> = null;
        if (options.extractTraces) {
          if (!llm || !frameworkLibrary || !traceStore) {
            traceResult = null;
          } else {
            traceResult = await extractReasoningTrace(
              segment,
              llm,
              frameworkLibrary,
              beliefGraph
            );
            if (traceResult?.trace) {
              await traceStore.addTrace(
                {
                  ...traceResult.trace,
                  source_refs: [doc.path],
                },
                llm
              );
            }
          }
        }

        extractions.push({
          source: doc.path,
          segment: segment.text,
          extraction: extracted,
          reasoning_trace: traceResult?.trace ?? null,
        });

        if (options.updateBeliefs) {
          for (const belief of extracted.beliefs) {
            const confidence = normalizeConfidence(belief.confidence);
            await beliefGraph.addBelief(
              {
                statement: belief.belief,
                domain_tags: belief.domain,
                confidence,
                first_stated: belief.stated_date,
                last_updated: belief.stated_date,
                source_refs: [doc.path],
              },
              llm
            );
          }
        }
      }
    }

    if (options.updateBeliefs) {
      console.error("Saving belief graph...");
      await beliefGraph.save();
    }

    if (options.extractTraces && traceStore) {
      console.error("Saving reasoning trace store...");
      await traceStore.save();
    }

    console.error(`\nExtraction complete. Total ${extractions.length} segments processed.`);
    console.log(JSON.stringify(extractions, null, 2));
  });

program
  .command("list-beliefs")
  .option("--beliefs <path>", "belief graph JSON", DEFAULT_BELIEFS)
  .action(async (options) => {
    const beliefGraph = await BeliefGraph.load(options.beliefs);
    const payload = beliefGraph.nodes.map((node) => ({
      id: node.id,
      statement: node.statement,
      confidence: node.confidence,
      domain_tags: node.domain_tags,
    }));
    console.log(JSON.stringify(payload, null, 2));
  });

program
  .command("tag-beliefs")
  .option("--beliefs <path>", "belief graph JSON", DEFAULT_BELIEFS)
  .option("--dry-run", "preview tags without saving")
  .option("--batch-size <size>", "beliefs per LLM request", "20")
  .action(async (options) => {
    const llm = buildLLM(true);
    if (!llm) {
      console.error("Tagging requires an LLM. Set OPENAI_API_KEY.");
      process.exit(1);
    }

    const beliefGraph = await BeliefGraph.load(options.beliefs);
    const untagged = beliefGraph.nodes.filter(
      (node) => !Array.isArray(node.domain_tags) || node.domain_tags.length === 0
    );

    if (untagged.length === 0) {
      console.error("All beliefs already have domain tags.");
      return;
    }

    const batchSize = Math.max(1, Number.parseInt(options.batchSize, 10) || 20);
    const taxonomySet = new Set(BELIEF_DOMAIN_TAXONOMY);
    const resultsSchema = z.record(z.array(z.string()));
    const failures: Array<{ id: string; statement: string; reason: string }> = [];
    let updated = 0;

    console.error(
      `Tagging ${untagged.length} belief(s) in batches of ${batchSize}...`
    );

    for (let i = 0; i < untagged.length; i += batchSize) {
      const batch = untagged.slice(i, i + batchSize);
      const prompt = `Given this taxonomy:\n${BELIEF_DOMAIN_TAXONOMY.join(
        ", "
      )}\n\nFor each belief statement, assign 2-5 tags that best capture the domains. Use only tags from the taxonomy.\n\nBeliefs:\n${batch
        .map((node, index) => `${index + 1}. "${node.statement}"`)
        .join("\n")}\n\nReturn JSON: { "1": ["ai", "economics"], "2": ["regulation", "policy"], ... }`;

      const response = await llm.generateJSON({
        system:
          "You label belief statements with domain tags from a fixed taxonomy.",
        user: prompt,
        schema: resultsSchema,
      });

      for (let j = 0; j < batch.length; j += 1) {
        const node = batch[j];
        const key = String(j + 1);
        const rawTags = response[key];
        if (!Array.isArray(rawTags)) {
          failures.push({
            id: node.id,
            statement: node.statement,
            reason: "missing tags",
          });
          continue;
        }

        const normalized = rawTags
          .filter((tag) => typeof tag === "string")
          .map((tag) => tag.trim().toLowerCase())
          .filter((tag) => tag.length > 0);
        const unique = Array.from(new Set(normalized));
        const hasUnknown = unique.some((tag) => !taxonomySet.has(tag));
        if (unique.length < 2 || unique.length > 5 || hasUnknown) {
          failures.push({
            id: node.id,
            statement: node.statement,
            reason: `invalid tags: ${JSON.stringify(unique)}`,
          });
          continue;
        }

        node.domain_tags = unique;
        updated += 1;
      }

      console.error(
        `Processed ${Math.min(i + batch.length, untagged.length)}/${
          untagged.length
        } beliefs`
      );
    }

    if (failures.length > 0) {
      console.error("Tagging failures:");
      for (const failure of failures) {
        console.error(`- ${failure.id}: ${failure.reason}`);
      }
    }

    if (options.dryRun) {
      console.error(
        `Dry run complete. Would update ${updated}/${untagged.length} beliefs.`
      );
      return;
    }

    console.error(`Saving ${updated} updated belief(s)...`);
    await beliefGraph.save();
    console.error("Tagging complete.");
  });

program
  .command("list-frameworks")
  .option("--frameworks <path>", "framework library JSON", DEFAULT_FRAMEWORKS)
  .action(async (options) => {
    const library = await FrameworkLibrary.load(options.frameworks);
    const payload = library.frameworks.map((framework) => ({
      id: framework.id,
      name: framework.name,
      triggers: framework.trigger_conditions,
    }));
    console.log(JSON.stringify(payload, null, 2));
  });

program
  .command("list-analogies")
  .option("--analogies <path>", "analogy index JSON", DEFAULT_ANALOGIES)
  .action(async (options) => {
    const index = await AnalogyIndex.load(options.analogies);
    const payload = index.analogies.map((analogy) => ({
      id: analogy.id,
      source_domain: analogy.source_domain,
      target_domain: analogy.target_domain,
      lessons: analogy.lessons,
    }));
    console.log(JSON.stringify(payload, null, 2));
  });

program
  .command("list-traces")
  .option("--traces <path>", "reasoning trace store JSON", DEFAULT_TRACES)
  .option("--framework <id>", "filter by framework id")
  .option("--belief <id>", "filter by belief id")
  .action(async (options) => {
    const store = await ReasoningTraceStore.load(options.traces);
    let traces = store.traces;
    if (options.framework) {
      traces = store.getByFramework(options.framework);
    }
    if (options.belief) {
      traces = traces.filter((trace) =>
        trace.beliefs_invoked.includes(options.belief)
      );
    }
    const payload = traces.map((trace) => ({
      id: trace.id,
      input_context: trace.input_context,
      conclusion: trace.conclusion,
      frameworks: trace.framework_selection?.chosen ?? [],
      beliefs_invoked: trace.beliefs_invoked,
      confidence: trace.confidence,
      extracted_at: trace.extracted_at,
    }));
    console.log(JSON.stringify(payload, null, 2));
  });

program
  .command("repl")
  .option("--frameworks <path>", "framework library JSON", DEFAULT_FRAMEWORKS)
  .option("--beliefs <path>", "belief graph JSON", DEFAULT_BELIEFS)
  .option("--analogies <path>", "analogy index JSON", DEFAULT_ANALOGIES)
  .option("--traces <path>", "reasoning trace store JSON", DEFAULT_TRACES)
  .option("--no-llm", "skip LLM synthesis")
  .action(async (options) => {
    const llm = buildLLM(options.llm);
    const [frameworks, beliefs, analogies, traces] = await Promise.all([
      FrameworkLibrary.load(options.frameworks),
      BeliefGraph.load(options.beliefs),
      AnalogyIndex.load(options.analogies),
      ReasoningTraceStore.load(options.traces),
    ]);

    const orchestrator = new Orchestrator(
      llm,
      frameworks,
      beliefs,
      analogies,
      traces
    );
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      prompt: "marc> ",
    });

    rl.prompt();
    rl.on("line", async (line) => {
      const query = line.trim();
      if (!query) {
        rl.prompt();
        return;
      }
      if (query === ":q" || query === ":quit") {
        rl.close();
        return;
      }
      const result = await orchestrator.reason(query);
      console.log(JSON.stringify(result, null, 2));
      rl.prompt();
    });
  });

program.parseAsync(process.argv).catch((error) => {
  console.error(error);
  process.exit(1);
});

function buildLLM(enabled: boolean | undefined): LLMClient | undefined {
  if (enabled === false) {
    return undefined;
  }
  try {
    return new LLMClient();
  } catch {
    return undefined;
  }
}

function normalizeConfidence(value: unknown): number {
  if (typeof value === "number") {
    return Math.min(Math.max(value, 0), 1);
  }
  if (typeof value === "string") {
    const normalized = value.toLowerCase();
    if (normalized.includes("high")) {
      return 0.8;
    }
    if (normalized.includes("medium")) {
      return 0.6;
    }
    if (normalized.includes("low")) {
      return 0.4;
    }
  }
  return 0.5;
}
