import express from "express";
import { protectRoute } from "../middleware/auth.js";
import { analyzeMessage, checkNlpServiceHealth } from "../services/nlpClient.js";

const nlpRouter = express.Router();

// Lets the frontend (or ops/monitoring) check whether the Python NLP
// service is currently reachable from the Node backend.
nlpRouter.get("/health", async (req, res) => {
    const health = await checkNlpServiceHealth();
    res.json({ success: true, ...health });
});

// Runs the real Adaptive Context Activation + model-tier cascade pipeline
// on an arbitrary (text, history) pair without persisting anything. Used
// by the Algorithm Showcase to demonstrate the real pipeline - both on
// the two canonical example workflows and on any custom text a user
// types in - not a canned/mocked simulation.
nlpRouter.post("/simulate", protectRoute, async (req, res) => {
    const { text, history } = req.body;
    if (!text || typeof text !== "string" || !text.trim()) {
        return res.json({ success: false, message: "text is required" });
    }
    const safeHistory = Array.isArray(history) ? history.filter((h) => typeof h === "string" && h.trim()) : [];

    const result = await analyzeMessage(text, safeHistory);
    if (!result) {
        return res.json({ success: false, message: "NLP service is unavailable" });
    }
    res.json({ success: true, result });
});

export default nlpRouter;
