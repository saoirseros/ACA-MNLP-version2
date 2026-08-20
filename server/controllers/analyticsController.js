import { getConversationAnalytics } from "../services/conversationAnalyticsService.js";

// Returns conversation-level analytics (Phase 8) for the 1:1 conversation
// between the logged-in user and the given other user id: aggregated
// sentiment/emotion/toxicity stats, context usage stats, and a freshly
// generated summary/topic for the conversation.
export const getConversationAnalyticsHandler = async (req, res) => {
    try {
        const myId = req.user._id;
        const { id: otherUserId } = req.params;

        const analytics = await getConversationAnalytics(myId, otherUserId);

        res.json({ success: true, analytics });
    } catch (error) {
        console.log(error.message);
        res.json({ success: false, message: error.message });
    }
};
