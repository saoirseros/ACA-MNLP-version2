import mongoose from "mongoose";

// Stores the per-message NLP analysis results. Kept as a separate
// collection (rather than embedding on Message) so the NLP layer can be
// extended independently and a missing/failed analysis never affects the
// core Message document.
const messageAnalysisSchema = new mongoose.Schema({
    messageId: { type: mongoose.Schema.Types.ObjectId, ref: "Message", required: true, unique: true },
    conversationId: { type: String, required: true, index: true },
    // Copied from the underlying Message's createdAt so conversation-level
    // aggregation can sort/trend analyses in actual message order without
    // an extra join/lookup.
    messageCreatedAt: { type: Date, required: true, index: true },
    sentiment: {
        label: { type: String },
        confidence: { type: Number },
    },
    emotion: {
        label: { type: String },
        confidence: { type: Number },
    },
    toxicity: {
        label: { type: String },
        confidence: { type: Number },
        category: { type: String, default: null },
    },
    // Adaptive Context Activation (Phase 6/7) outcome for this message:
    // how much conversational context was actually used, how confident
    // that decision was, and how many prior messages were selected.
    contextLevel: { type: String, enum: ["low", "medium", "high"], default: "low" },
    contextScore: { type: Number },
    selectedContextMessages: { type: Number, default: 0 },
    // Per-module model name + latency, e.g. { sentiment: {...}, emotion: {...}, toxicity: {...} }
    processing: { type: mongoose.Schema.Types.Mixed },
    totalLatencyMs: { type: Number },
    status: { type: String, enum: ["completed", "failed", "unavailable"], default: "completed" },
}, { timestamps: true });

const MessageAnalysis = mongoose.model("MessageAnalysis", messageAnalysisSchema);

export default MessageAnalysis;
