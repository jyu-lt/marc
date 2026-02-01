import "dotenv/config";
import { Command } from "commander";
import path from "path";
import readline from "readline";
import { LLMClient } from "./llm/client.js";
import { FrameworkLibrary } from "./knowledge/frameworks.js";
import { BeliefGraph } from "./knowledge/beliefs.js";
import { AnalogyIndex } from "./knowledge/analogies.js";
import { Orchestrator } from "./inference/orchestrator.js";
import { loadCorpus, loadDocument } from "./ingestion/loader.js";
import { segmentDocument } from "./ingestion/segmenter.js";
import { extractFromSegment } from "./ingestion/extractor.js";

const DEFAULT_FRAMEWORKS = "data/frameworks.json";
const DEFAULT_BELIEFS = "data/beliefs.json";
const DEFAULT_ANALOGIES = "data/analogies.json";
const DEFAULT_CORPUS = "data/corpus";

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
  .option("--no-llm", "skip LLM synthesis")
  .action(async (query, options) => {
    const llm = buildLLM(options.llm);
    const [frameworks, beliefs, analogies] = await Promise.all([
      FrameworkLibrary.load(options.frameworks),
      BeliefGraph.load(options.beliefs),
      AnalogyIndex.load(options.analogies),
    ]);

    const orchestrator = new Orchestrator(llm, frameworks, beliefs, analogies);
    const result = await orchestrator.reason(query);
    console.log(JSON.stringify(result, null, 2));
  });

program
  .command("extract")
  .argument("[path]", "file or directory to extract", DEFAULT_CORPUS)
  .option("--beliefs <path>", "belief graph JSON", DEFAULT_BELIEFS)
  .option(
    "--method <method>",
    "segmentation method (paragraph|llm|markdown)",
    "paragraph"
  )
  .option("--update-beliefs", "merge extracted beliefs into belief graph")
  .option("--no-llm", "skip LLM extraction")
  .action(async (targetPath, options) => {
    const llm = buildLLM(options.llm);
    const absolutePath = path.resolve(targetPath);
    const stats = await import("fs").then((mod) =>
      mod.promises.stat(absolutePath)
    );
    const docs = stats.isDirectory()
      ? await loadCorpus(absolutePath)
      : [await loadDocument(absolutePath)];

    console.error(`Found ${docs.length} document(s) at ${absolutePath}`);

    const beliefGraph = await BeliefGraph.load(options.beliefs);
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

        extractions.push({
          source: doc.path,
          segment: segment.text,
          extraction: extracted,
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
  .command("repl")
  .option("--frameworks <path>", "framework library JSON", DEFAULT_FRAMEWORKS)
  .option("--beliefs <path>", "belief graph JSON", DEFAULT_BELIEFS)
  .option("--analogies <path>", "analogy index JSON", DEFAULT_ANALOGIES)
  .option("--no-llm", "skip LLM synthesis")
  .action(async (options) => {
    const llm = buildLLM(options.llm);
    const [frameworks, beliefs, analogies] = await Promise.all([
      FrameworkLibrary.load(options.frameworks),
      BeliefGraph.load(options.beliefs),
      AnalogyIndex.load(options.analogies),
    ]);

    const orchestrator = new Orchestrator(llm, frameworks, beliefs, analogies);
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
