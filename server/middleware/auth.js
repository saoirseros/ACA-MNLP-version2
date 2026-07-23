import User from "../models/User.js";
import jwt from "jsonwebtoken";

//to confirm user authentication, middleware to protect routes i.e. user is only routed if authenticated

export const protectRoute = async (req, res, next)=>{
    try {
        const token = req.headers.token;

        if (!token) {
            return res.json({
                success:false,
                message:"Unauthorized"
            });
        }

        const decoded = jwt.verify(token, process.env.JWT_SECRET)

        const user = await User.findById(decoded.userId).select("-password");

        if(!user) return res.json({success: false, message: "User not found" });

        req.user = user;
        next(); //executes the controller function

    } catch (error) {
        console.log(error.message);
        res.json({success: false, message: error.message });
    }
}