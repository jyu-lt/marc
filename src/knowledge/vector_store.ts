import * as fs from "node:fs/promises";
import * as path from "node:path";

const MAGIC = "VF01"; // Vector Format 01
const MAGIC_BUFFER = Buffer.from(MAGIC, "ascii");

export class VectorStore {
  /**
   * Loads the vector store from a binary file.
   * Format:
   * - Header: 4 bytes "VF01"
   * - Count: 4 bytes UInt32LE
   * - Records:
   *   - ID Length: 4 bytes UInt32LE
   *   - ID: UTF-8 string bytes
   *   - Vector: 1536 * 4 bytes (Float32LE)
   */
  async load(filePath: string): Promise<Map<string, Float32Array>> {
    const vectors = new Map<string, Float32Array>();

    try {
      const handle = await fs.open(filePath, "r");
      try {
        const stat = await handle.stat();
        const buffer = Buffer.alloc(stat.size);
        await handle.read(buffer, 0, stat.size, 0);

        let offset = 0;

        // Check Magic
        if (
          buffer.subarray(offset, offset + 4).toString("ascii") !== MAGIC
        ) {
          throw new Error("Invalid vector store format");
        }
        offset += 4;

        // Read Count
        const count = buffer.readUInt32LE(offset);
        offset += 4;

        for (let i = 0; i < count; i++) {
          // ID Length
          const idLen = buffer.readUInt32LE(offset);
          offset += 4;

          // ID
          const id = buffer.toString("utf8", offset, offset + idLen);
          offset += idLen;

          // Vector (assuming 1536 dimensions for now, but dynamic reading is safer)
          // Ideally we store dimension or rely on standard size.
          // For now, let's assume we read until end or structure dictates.
          // Wait, the plan said fixed size? Actually, let's store dimension per vector or globally.
          // Plan didn't specify dynamic size handling detail but standard OpenAI is 1536.
          // A robust format would store dimension.
          // Let's UPDATE the format on the fly to include dimension count per vector to be safe/flexible.
          // Updated Format Record:
          // [ID_LEN: 4][ID: bytes][DIM: 4][VECTOR: DIM * 4]

          const dim = buffer.readUInt32LE(offset);
          offset += 4;

          const byteLength = dim * 4;
          const vectorBuffer = buffer.subarray(offset, offset + byteLength);
          // Create a copy to ensure aligned access if needed and detach from large buffer
          const vector = new Float32Array(
            vectorBuffer.buffer.slice(
              vectorBuffer.byteOffset,
              vectorBuffer.byteOffset + vectorBuffer.byteLength
            )
          );
          
          offset += byteLength;

          vectors.set(id, vector);
        }
      } finally {
        await handle.close();
      }
    } catch (error: any) {
      if (error.code !== "ENOENT") {
        console.error("Failed to load vector store:", error);
        throw error;
      }
      // Return empty map if file doesn't exist
    }

    return vectors;
  }

  async save(filePath: string, vectors: Map<string, Float32Array>): Promise<void> {
    // Calculate size
    let size = 4 + 4; // Magic + Count
    for (const [id, vector] of vectors) {
      const idBytes = Buffer.byteLength(id, "utf8");
      size += 4 + idBytes + 4 + vector.length * 4; // ID_Len + ID + Dim + Vector
    }

    const buffer = Buffer.alloc(size);
    let offset = 0;

    // Header
    MAGIC_BUFFER.copy(buffer, offset);
    offset += 4;

    // Count
    buffer.writeUInt32LE(vectors.size, offset);
    offset += 4;

    for (const [id, vector] of vectors) {
      // ID Length
      const idBytes = Buffer.byteLength(id, "utf8");
      buffer.writeUInt32LE(idBytes, offset);
      offset += 4;

      // ID
      buffer.write(id, offset, "utf8");
      offset += idBytes;

      // Dimension
      buffer.writeUInt32LE(vector.length, offset);
      offset += 4;

      // Vector
      const float32Buffer = Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength);
      float32Buffer.copy(buffer, offset);
      offset += float32Buffer.length;
    }

    // Ensure dir exists
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, buffer);
  }
}
