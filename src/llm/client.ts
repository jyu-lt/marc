import OpenAI from "openai";
import type { ResponseFormatTextConfig } from "openai/resources/responses/responses";
import { z } from "zod";

export type LLMClientOptions = {
  apiKey?: string;
  model?: string;
  embeddingModel?: string;
  maxRetries?: number;
};

export type JSONSchemaFormat = {
  name: string;
  schema: Record<string, unknown>;
  description?: string;
  strict?: boolean;
};

export class LLMClient {
  private client: OpenAI;
  private model: string;
  private embeddingModel: string;
  private maxRetries: number;

  constructor(options: LLMClientOptions = {}) {
    const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error("Missing OPENAI_API_KEY");
    }
    this.client = new OpenAI({ apiKey });
    this.model = options.model ?? process.env.OPENAI_MODEL ?? "gpt-5.2";
    this.embeddingModel =
      options.embeddingModel ??
      process.env.OPENAI_EMBEDDING_MODEL ??
      "text-embedding-3-small";
    this.maxRetries = options.maxRetries ?? 2;
  }

  async generateText(system: string, user: string): Promise<string> {
    const response = await this.client.responses.create({
      model: this.model,
      input: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    });
    return response.output_text ?? "";
  }

  async generateJSON<S extends z.ZodTypeAny>(params: {
    system: string;
    user: string;
    schema: S;
    jsonSchema?: JSONSchemaFormat;
  }): Promise<z.infer<S>>;
  async generateJSON<T>(params: {
    system: string;
    user: string;
    schema?: z.ZodType<T>;
    jsonSchema?: JSONSchemaFormat;
  }): Promise<T> {
    const responseFormat: ResponseFormatTextConfig = params.jsonSchema
      ? {
          type: "json_schema",
          name: params.jsonSchema.name,
          description: params.jsonSchema.description,
          schema: params.jsonSchema.schema,
          strict: params.jsonSchema.strict ?? true,
        }
      : {
          type: "json_object",
        };

    const systemMessage = `${params.system}\n\nYou must respond with JSON.`;

    return this.withRetry(async () => {
      const response = await this.client.responses.create({
        model: this.model,
        input: [
          { role: "system", content: systemMessage },
          { role: "user", content: params.user },
        ],
        text: {
          format: responseFormat,
        },
      });

      const outputText = response.output_text ?? "";
      const parsed = JSON.parse(outputText) as T;
      if (params.schema) {
        return params.schema.parse(parsed);
      }
      return parsed;
    });
  }

  async embed(input: string | string[]): Promise<number[] | number[][]> {
    const response = await this.client.embeddings.create({
      model: this.embeddingModel,
      input,
      encoding_format: "float",
    });
    if (Array.isArray(input)) {
      return response.data.map((item) => item.embedding);
    }
    return response.data[0]?.embedding ?? [];
  }

  private async withRetry<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: unknown = null;
    for (let attempt = 0; attempt <= this.maxRetries; attempt += 1) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        if (attempt === this.maxRetries) {
          break;
        }
        await this.sleep(400 * Math.pow(2, attempt));
      }
    }
    throw lastError instanceof Error
      ? lastError
      : new Error("LLM request failed");
  }

  private async sleep(ms: number): Promise<void> {
    await new Promise((resolve) => setTimeout(resolve, ms));
  }
}
