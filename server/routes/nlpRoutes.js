import express from "express";
import { checkNlpServiceHealth } from "../services/nlpClient.js";

const nlpRouter = express.Router();

// Lets the frontend (or ops/monitoring) check whether the Python NLP
// service is currently reachable from the Node backend.
nlpRouter.get("/health", async (req, res) => {
    const health = await checkNlpServiceHealth();
    res.json({ success: true, ...health });
});

export default nlpRouter;
