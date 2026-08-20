import Message from "../models/Message.js";
import MessageAnalysis from "../models/MessageAnalysis.js";
import { analyzeMessage } from "./nlpClient.js";
import { getConversationId } from "../lib/utils.js";
import { io, userSocketMap } from "../server.js";

// How many recent messages (max) to offer Adaptive Context Activation as
// candidate history. ACA itself decides how much of this is actually
// used - this limit just bounds the size of what we fetch/send.
const HISTORY_LIMIT = 15;

// Fetch the most recent text messages between the two participants of
// `message`, oldest first, excluding `message` itself and any image-only
// messages (which have no text to analyze).
const fetchRecentHistoryTexts = async (message) => {
    const recentMessages = await Message.find({
        _id: { $ne: message._id },
        $or: [
            { senderId: message.senderId, receiverId: message.receiverId },
            { senderId: message.receiverId, receiverId: message.senderId },
        ],
    })
        .sort({ createdAt: -1 })
        .limit(HISTORY_LIMIT)
        .select("text createdAt");

    return recentMessages
        .reverse() // oldest first, matching Adaptive Context Activation's expected input order
        .map((m) => m.text)
        .filter((text) => typeof text === "string" && text.trim().length > 0);
};

// Runs NLP analysis for a message in the background and emits the result
// over Socket.io once ready. This must never throw or block normal chat -
// if the NLP service is down or a request fails, we simply skip emitting
// an analysis update (spec: "NORMAL CHAT MUST STILL WORK").
export const analyzeMessageAsync = async (message) => {
    if (!message.text) return; // Text-only messages are analyzed for now

    try {
        const history = await fetchRecentHistoryTexts(message);
        const result = await analyzeMessage(message.text, history);
        if (!result) return; // NLP service unavailable or the request failed

        const conversationId = getConversationId(message.senderId, message.receiverId);

        const analysis = await MessageAnalysis.findOneAndUpdate(
            { messageId: message._id },
            {
                messageId: message._id,
                conversationId,
                messageCreatedAt: message.createdAt,
                sentiment: result.sentiment,
                emotion: result.emotion,
                toxicity: result.toxicity,
                processing: result.processing,
                totalLatencyMs: result.totalLatencyMs,
                contextLevel: result.contextLevel,
                contextScore: result.contextScore,
                selectedContextMessages: result.selectedContextMessages,
                status: "completed",
            },
            { upsert: true, new: true }
        );

        const payload = {
            messageId: message._id,
            conversationId,
            sentiment: analysis.sentiment,
            emotion: analysis.emotion,
            toxicity: analysis.toxicity,
            processing: analysis.processing,
            totalLatencyMs: analysis.totalLatencyMs,
            contextLevel: analysis.contextLevel,
            contextScore: analysis.contextScore,
            selectedContextMessages: analysis.selectedContextMessages,
        };

        // Notify both participants so either side of the chat sees the
        // analysis update in real time.
        [message.senderId, message.receiverId].forEach((userId) => {
            const socketId = userSocketMap[userId];
            if (socketId) io.to(socketId).emit("messageAnalysis", payload);
        });
    } catch (error) {
        console.log("Message analysis pipeline failed:", error.message);
    }
};
