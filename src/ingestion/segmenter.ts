import { z } from "zod";
import { newId } from "../utils/ids.js";
import type { CorpusDocument } from "./loader.js";
import type { LLMClient } from "../llm/client.js";

export type Segment = {
  id: string;
  docId: string;
  sourcePath: string;
  text: string;
  startIndex: number;
  endIndex: number;
};

export type SegmenterOptions = {
  method?: "paragraph" | "llm" | "markdown";
  maxChars?: number;
  llm?: LLMClient;
};

const SegmentSchema = z.object({
  segments: z.array(
    z.object({
      text: z.string().min(1),
      start_char: z.number().min(0),
      end_char: z.number().min(0),
    })
  ),
});

export async function segmentDocument(
  doc: CorpusDocument,
  options: SegmenterOptions = {}
): Promise<Segment[]> {
  const method = options.method ?? "paragraph";
  const maxChars = options.maxChars ?? 1200;

  if (method === "markdown") {
    return segmentByMarkdown(doc);
  }

  if (method === "llm" && options.llm && doc.content.length <= 8000) {
    try {
      return await segmentWithLLM(doc, options.llm);
    } catch {
      // Fall back to paragraph segmentation.
    }
  }

  return segmentByParagraph(doc, maxChars);
}

function segmentByParagraph(doc: CorpusDocument, maxChars: number): Segment[] {
  const blocks = doc.content
    .split(/\n\s*\n/)
    .map((block) => block.trim())
    .filter(Boolean);

  const segments: Segment[] = [];
  let cursor = 0;

  for (const block of blocks) {
    const startIndex = doc.content.indexOf(block, cursor);
    const endIndex = startIndex + block.length;
    cursor = endIndex;

    if (block.length <= maxChars) {
      segments.push({
        id: newId(),
        docId: doc.id,
        sourcePath: doc.path,
        text: block,
        startIndex,
        endIndex,
      });
      continue;
    }

    let offset = 0;
    while (offset < block.length) {
      const slice = block.slice(offset, offset + maxChars);
      const sliceStart = startIndex + offset;
      const sliceEnd = sliceStart + slice.length;
      segments.push({
        id: newId(),
        docId: doc.id,
        sourcePath: doc.path,
        text: slice,
        startIndex: sliceStart,
        endIndex: sliceEnd,
      });
      offset += maxChars;
    }
  }

  return segments;
}

function segmentByMarkdown(doc: CorpusDocument): Segment[] {
  const lines = doc.content.split("\n");
  const segments: Segment[] = [];

  let currentSection: string[] = [];
  let currentStartIndex = 0;
  let currentLineIndex = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Check if this is an H2 header (section boundary)
    if (line.match(/^##\s+\w/) && currentSection.length > 0) {
      // Save the previous section
      const text = currentSection.join("\n").trim();
      if (text) {
        segments.push({
          id: newId(),
          docId: doc.id,
          sourcePath: doc.path,
          text,
          startIndex: currentStartIndex,
          endIndex: currentStartIndex + text.length,
        });
      }

      // Start new section
      currentSection = [line];
      currentStartIndex = doc.content.indexOf(line, currentLineIndex);
      currentLineIndex = currentStartIndex;
    } else {
      if (currentSection.length === 0) {
        currentStartIndex = doc.content.indexOf(line, currentLineIndex);
        currentLineIndex = currentStartIndex;
      }
      currentSection.push(line);
    }
  }

  // Don't forget the last section
  if (currentSection.length > 0) {
    const text = currentSection.join("\n").trim();
    if (text) {
      segments.push({
        id: newId(),
        docId: doc.id,
        sourcePath: doc.path,
        text,
        startIndex: currentStartIndex,
        endIndex: currentStartIndex + text.length,
      });
    }
  }

  return segments;
}

async function segmentWithLLM(
  doc: CorpusDocument,
  llm: LLMClient
): Promise<Segment[]> {
  const prompt = `Split the following text into coherent argument units. Return JSON with a segments array. Each segment must include text, start_char, and end_char indices relative to the original text.\n\nTEXT:\n${doc.content}`;

  const parsed = await llm.generateJSON({
    system: "You are a segmentation assistant.",
    user: prompt,
    schema: SegmentSchema,
  });

  return parsed.segments.map((segment) => ({
    id: newId(),
    docId: doc.id,
    sourcePath: doc.path,
    text: segment.text,
    startIndex: segment.start_char,
    endIndex: segment.end_char,
  }));
}
