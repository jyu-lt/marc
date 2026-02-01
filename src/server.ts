import "dotenv/config";
import express from "express";
import cors from "cors";
import { LLMClient } from "./llm/client.js";
import { FrameworkLibrary } from "./knowledge/frameworks.js";
import { BeliefGraph } from "./knowledge/beliefs.js";
import { AnalogyIndex } from "./knowledge/analogies.js";
import { Orchestrator } from "./inference/orchestrator.js";

const DEFAULT_FRAMEWORKS = "data/frameworks.json";
const DEFAULT_BELIEFS = "data/beliefs.json";
const DEFAULT_ANALOGIES = "data/analogies.json";

async function startServer() {
  const app = express();
  app.use(cors());
  app.use(express.json());

  const port = process.env.PORT || 3001;

  console.log("Loading knowledge bases...");
  const [frameworks, beliefs, analogies] = await Promise.all([
    FrameworkLibrary.load(DEFAULT_FRAMEWORKS),
    BeliefGraph.load(DEFAULT_BELIEFS),
    AnalogyIndex.load(DEFAULT_ANALOGIES),
  ]);

  const llm = new LLMClient();
  const orchestrator = new Orchestrator(llm, frameworks, beliefs, analogies);

  app.post("/api/ask", async (req, res) => {
    try {
      const { query } = req.body;
      if (!query) {
        return res.status(400).json({ error: "Query is required" });
      }

      console.log(`Processing query: ${query}`);
      const result = await orchestrator.reason(query);
      res.json(result);
    } catch (error) {
      console.error("Error processing query:", error);
      res.status(500).json({ error: "Internal server error" });
    }
  });

  app.get("/api/health", (req, res) => {
    res.json({ status: "ok" });
  });

  app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
  });
}

startServer().catch((error) => {
  console.error("Failed to start server:", error);
  process.exit(1);
});
