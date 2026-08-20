import jwt from "jsonwebtoken";

//fucntion to generate a token for user
export const generateToken = (userId)=>{
    const token = jwt.sign({userId}, process.env.JWT_SECRET);
    return token;
}

// Derive a stable, order-independent conversation id for a 1:1 chat from
// the two participant user ids. Used to group per-message NLP analysis
// results into conversation-level analytics.
export const getConversationId = (userIdA, userIdB)=>{
    return [userIdA.toString(), userIdB.toString()].sort().join("_");
}