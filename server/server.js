import express from "express";
import "dotenv/config";
import cors from "cors";
import http from "http";
import { connectDB } from "./lib/db.js";
import userRouter from "./routes/userRoutes.js";
import messageRouter from "./routes/messageRoutes.js";
import { Server } from "socket.io";

import cloudinary from "./lib/cloudinary.js";

//creating express app using http server
const app = express();
const server = http.createServer(app)

//initializing socket.io server
export const io = new Server(server, {
    cors: {origin: "*"}
})

//store online users
export const userSocketMap = {}; //{userId: socketId}

//socket.io connection handler
io.on("connection", (socket)=>{
    const userId = socket.handshake.query.userId;
    console.log("User Connected", userId);

    if(userId) userSocketMap[userId] = socket.id;

    //emit online users to all connected client
    io.emit("getOnlineUsers", Object.keys(userSocketMap));

    socket.on("disconnect", ()=>{
        console.log("User Disconnected", userId);
        delete userSocketMap[userId];
        io.emit("getOnlineUsers", Object.keys(userSocketMap))
    })
})

//middleware setup
app.use(express.json({limit:"4mb"}));
app.use(cors());


//Routes setup
app.use("/api/status", (req,res)=> res.send("Server is live"));
app.use("/api/auth", userRouter);
app.use("/api/messages", messageRouter)

//Connect to mongodb
await connectDB();

const PORT = process.env.PORT || 5000;
server.listen(PORT, ()=>console.log("Server is running on PORT: "+PORT));

//test for cloudinary
app.get("/test-cloudinary", async (req, res) => {
    try {
        const result = await cloudinary.uploader.upload(
            "https://res.cloudinary.com/demo/image/upload/sample.jpg"
        );

        res.json(result);
    } catch (err) {
        console.log(JSON.stringify(err, null, 2));
        res.json(err);
    }
});
