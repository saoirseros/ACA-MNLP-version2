import axios from "axios";

// Base URL of the Python NLP service. Left unset in an environment where
// NLP analysis is not desired/available - callers must handle that case.
const NLP_SERVICE_URL = process.env.NLP_SERVICE_URL;
const REQUEST_TIMEOUT_MS = 8000;
// The combined /analyze/message call runs 3 models; the very first call
// after the service restarts also has to load each model into memory
// (mitigated by a background warm-up in the Python service, but this is
// a generous safety net in case a message arrives before warm-up finishes).
const MESSAGE_ANALYSIS_TIMEOUT_MS = 30000;

const nlpAxios = axios.create({
    baseURL: NLP_SERVICE_URL,
    timeout: REQUEST_TIMEOUT_MS,
});

// Check whether the Python NLP service is reachable. Never throws.
export const checkNlpServiceHealth = async () => {
    if (!NLP_SERVICE_URL) return { available: false, message: "NLP_SERVICE_URL is not configured" };
    try {
        const { data } = await nlpAxios.get("/health");
        return { available: true, ...data };
    } catch (error) {
        return { available: false, message: error.message };
    }
};

// Analyze the sentiment of a single message. Returns null (never throws)
// if the NLP service is unavailable or the request fails, so normal chat
// is never blocked or broken by NLP downtime.
export const analyzeSentiment = async (text) => {
    if (!NLP_SERVICE_URL) return null;
    try {
        const { data } = await nlpAxios.post("/analyze/sentiment", { text });
        return data;
    } catch (error) {
        console.log("NLP sentiment analysis failed:", error.message);
        return null;
    }
};

// Run every currently available NLP module (sentiment, emotion, toxicity)
// on a single message in one HTTP call, using Adaptive Context Activation
// to decide how much of `history` (if any) is actually used. Returns null
// (never throws) if the NLP service is unavailable or the request fails.
//
// `history` should be the preceding conversation messages, oldest first,
// current message excluded.
export const analyzeMessage = async (text, history = []) => {
    if (!NLP_SERVICE_URL) return null;
    try {
        const { data } = await nlpAxios.post(
            "/analyze/message",
            { text, history },
            { timeout: MESSAGE_ANALYSIS_TIMEOUT_MS }
        );
        return data;
    } catch (error) {
        console.log("NLP message analysis failed:", error.message);
        return null;
    }
};

// Summarize a whole conversation (ordered list of message texts, oldest
// first). Used by conversation-level analytics (Phase 8), not per-message.
// Returns null (never throws) if unavailable or the request fails.
export const summarizeConversation = async (messages) => {
    if (!NLP_SERVICE_URL || messages.length === 0) return null;
    try {
        const { data } = await nlpAxios.post(
            "/analyze/summarize",
            { messages },
            { timeout: MESSAGE_ANALYSIS_TIMEOUT_MS }
        );
        return data;
    } catch (error) {
        console.log("NLP conversation summarization failed:", error.message);
        return null;
    }
};

// Extract representative topic keywords from a whole conversation. Used
// by conversation-level analytics (Phase 8), not per-message. Returns
// null (never throws) if unavailable or the request fails.
export const extractTopics = async (messages) => {
    if (!NLP_SERVICE_URL || messages.length === 0) return null;
    try {
        const { data } = await nlpAxios.post("/analyze/topics", { messages });
        return data;
    } catch (error) {
        console.log("NLP topic extraction failed:", error.message);
        return null;
    }
};
