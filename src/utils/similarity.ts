export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) {
    return 0;
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) {
    return 0;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function averageEmbedding(vectors: number[][]): number[] {
  if (vectors.length === 0) {
    return [];
  }
  const length = vectors[0].length;
  const sum = new Array<number>(length).fill(0);
  for (const vec of vectors) {
    for (let i = 0; i < length; i += 1) {
      sum[i] += vec[i];
    }
  }
  return sum.map((value) => value / vectors.length);
}
