import Message from "../models/Message.js";
import MessageAnalysis from "../models/MessageAnalysis.js";
import { extractTopics, summarizeConversation } from "./nlpClient.js";
import { getConversationId } from "../lib/utils.js";

// How many of the most recent conversation messages to feed into
// summarization/topic extraction (these operate on the raw conversation,
// not the stored per-message analyses).
const RECENT_MESSAGES_FOR_SUMMARY = 20;

// Need at least this many analyzed messages before attempting a
// directional sentiment-trend statement - below this, there just isn't
// enough data to say anything meaningful.
const MIN_MESSAGES_FOR_TREND = 4;

// Minimum shift in average signed sentiment (see signedSentimentScore)
// between the first and second half of the conversation to call it a
// "trend" rather than noise.
const TREND_THRESHOLD = 0.3;

// Converts a sentiment result into a signed score (+confidence for
// positive, -confidence for negative), used only internally to detect a
// directional trend - never displayed to the user as a "score".
const signedSentimentScore = (sentiment) => {
    if (!sentiment) return 0;
    return sentiment.label === "positive" ? sentiment.confidence : -sentiment.confidence;
};

const average = (numbers) => (numbers.length === 0 ? null : numbers.reduce((a, b) => a + b, 0) / numbers.length);

const mostFrequent = (labels) => {
    if (labels.length === 0) return null;
    const counts = {};
    for (const label of labels) counts[label] = (counts[label] || 0) + 1;
    return Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];
};

// Compares the average sentiment of the first vs second half of the
// conversation and only emits a directional statement when the shift is
// large enough to be meaningful - per the spec: "Only generate such
// higher-level statements when the underlying data supports them."
const buildSentimentTrendInsight = (analyses) => {
    if (analyses.length < MIN_MESSAGES_FOR_TREND) {
        return "Not enough analyzed messages yet to determine a sentiment trend.";
    }

    const scores = analyses.map((a) => signedSentimentScore(a.sentiment));
    const mid = Math.floor(scores.length / 2);
    const firstHalfAvg = average(scores.slice(0, mid));
    const secondHalfAvg = average(scores.slice(mid));
    const shift = secondHalfAvg - firstHalfAvg;

    if (shift <= -TREND_THRESHOLD) {
        return "Conversation sentiment has become more negative over time.";
    }
    if (shift >= TREND_THRESHOLD) {
        return "Conversation sentiment has become more positive over time.";
    }
    return "Conversation sentiment has remained relatively stable.";
};

// Computes conversation-level analytics for the 1:1 conversation between
// `myId` and `otherUserId`: aggregates already-stored per-message
// analyses, and (best-effort, may be null if the NLP service is down)
// requests a fresh summary/topic extraction over the most recent
// messages, since those operate on raw conversation text rather than
// per-message analysis records.
export const getConversationAnalytics = async (myId, otherUserId) => {
    const conversationId = getConversationId(myId, otherUserId);

    const analyses = await MessageAnalysis.find({ conversationId, status: "completed" })
        .sort({ messageCreatedAt: 1 })
        .lean();

    const messagesAnalyzed = analyses.length;

    const sentimentLabels = analyses.map((a) => a.sentiment?.label).filter(Boolean);
    const emotionLabels = analyses.map((a) => a.emotion?.label).filter(Boolean);
    const toxicCount = analyses.filter((a) => a.toxicity?.label === "toxic").length;

    const sentimentConfidences = analyses.map((a) => a.sentiment?.confidence).filter((v) => typeof v === "number");
    const emotionConfidences = analyses.map((a) => a.emotion?.confidence).filter((v) => typeof v === "number");
    const totalLatencies = analyses.map((a) => a.totalLatencyMs).filter((v) => typeof v === "number");

    const contextLevelCounts = { low: 0, medium: 0, high: 0 };
    analyses.forEach((a) => {
        if (a.contextLevel && contextLevelCounts[a.contextLevel] !== undefined) {
            contextLevelCounts[a.contextLevel] += 1;
        }
    });
    const contextScores = analyses.map((a) => a.contextScore).filter((v) => typeof v === "number");
    const selectedContextCounts = analyses.map((a) => a.selectedContextMessages).filter((v) => typeof v === "number");

    const emotionDistribution = emotionLabels.reduce((acc, label) => {
        acc[label] = (acc[label] || 0) + 1;
        return acc;
    }, {});

    const sentimentProgression = analyses.map((a) => ({
        messageId: a.messageId,
        label: a.sentiment?.label,
        confidence: a.sentiment?.confidence,
        timestamp: a.messageCreatedAt,
    }));

    // Fetch the most recent text messages (regardless of whether they've
    // been individually analyzed yet) to generate a fresh summary/topic.
    const recentMessages = await Message.find({
        $or: [
            { senderId: myId, receiverId: otherUserId },
            { senderId: otherUserId, receiverId: myId },
        ],
    })
        .sort({ createdAt: -1 })
        .limit(RECENT_MESSAGES_FOR_SUMMARY)
        .select("text");

    const recentTexts = recentMessages
        .reverse() // oldest first
        .map((m) => m.text)
        .filter((text) => typeof text === "string" && text.trim().length > 0);

    const [summaryResult, topicResult] = await Promise.all([
        summarizeConversation(recentTexts),
        extractTopics(recentTexts),
    ]);

    return {
        conversationId,
        messagesAnalyzed,
        sentiment: {
            overall: mostFrequent(sentimentLabels),
            averageConfidence: average(sentimentConfidences),
            progression: sentimentProgression,
        },
        emotion: {
            dominant: mostFrequent(emotionLabels),
            averageConfidence: average(emotionConfidences),
            distribution: emotionDistribution,
        },
        toxicity: {
            toxicMessageCount: toxicCount,
            frequency: messagesAnalyzed > 0 ? toxicCount / messagesAnalyzed : 0,
        },
        topic: topicResult ? topicResult.topic : null,
        summary: summaryResult ? summaryResult.summary : null,
        context: {
            levelCounts: contextLevelCounts,
            averageScore: average(contextScores),
            averageSelectedMessages: average(selectedContextCounts),
        },
        processing: {
            averageTotalLatencyMs: average(totalLatencies),
        },
        insight: buildSentimentTrendInsight(analyses),
    };
};
