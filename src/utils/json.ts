import { promises as fs } from "fs";

export async function readJsonFile<T>(path: string, fallback: T): Promise<T> {
  try {
    const raw = await fs.readFile(path, "utf-8");
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export async function writeJsonFile(path: string, data: unknown): Promise<void> {
  const json = JSON.stringify(data, null, 2);
  await fs.writeFile(path, json, "utf-8");
}
