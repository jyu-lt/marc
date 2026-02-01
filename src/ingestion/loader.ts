import { promises as fs } from "fs";
import path from "path";
import { newId } from "../utils/ids.js";

export type CorpusDocument = {
  id: string;
  path: string;
  type: "txt" | "md" | "json";
  content: string;
  metadata: Record<string, unknown>;
};

const SUPPORTED_EXTENSIONS = new Set([".txt", ".md", ".json"]);

export async function loadCorpus(dir: string): Promise<CorpusDocument[]> {
  const entries = await listFiles(dir);
  const docs: CorpusDocument[] = [];
  for (const filePath of entries) {
    const ext = path.extname(filePath).toLowerCase();
    if (!SUPPORTED_EXTENSIONS.has(ext)) {
      continue;
    }
    docs.push(await loadDocument(filePath));
  }
  return docs;
}

export async function loadDocument(filePath: string): Promise<CorpusDocument> {
  const ext = path.extname(filePath).toLowerCase();
  const raw = await fs.readFile(filePath, "utf-8");
  let content = raw;
  let metadata: Record<string, unknown> = {};

  if (ext === ".json") {
    try {
      const parsed = JSON.parse(raw) as Record<string, unknown>;
      if (typeof parsed.text === "string") {
        content = parsed.text;
      } else if (Array.isArray(parsed.segments)) {
        content = parsed.segments
          .map((segment) =>
            typeof segment === "object" && segment && "text" in segment
              ? String((segment as { text?: string }).text ?? "")
              : ""
          )
          .filter(Boolean)
          .join("\n");
      } else {
        content = JSON.stringify(parsed);
      }
      metadata = parsed.metadata && typeof parsed.metadata === "object" ? (parsed.metadata as Record<string, unknown>) : {};
    } catch {
      content = raw;
    }
  }

  return {
    id: newId(),
    path: filePath,
    type: ext.replace(".", "") as CorpusDocument["type"],
    content,
    metadata
  };
}

async function listFiles(dir: string): Promise<string[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await listFiles(fullPath)));
    } else {
      files.push(fullPath);
    }
  }
  return files;
}
