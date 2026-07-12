import { useContext } from "react";
import { useState } from "react";
import { createContext } from "react";
import { AuthContext } from "./AuthContext";

export const ChatContext = createContext();

export const ChatProvider = ({ children })=>{

    const [messages, setMessages] = useState([]);
    const [users, setUsers] = useState([]);
    const [selectedUser, setSelectedUser] = useState(null);
    const [unseenMessages, setUnseenMessages] = useState({});

    const {socket, axios} = useContext(AuthContext);

    //function to get all users for sidebar
    const getUsers = async () =>{
        try {
            const { data } = await axios.get("/api/messages/users");
            if (data.success){
                setUsers(data.users)
                setUnseenMessages(data.unseenMessages)
            }
        } catch (error) {
            toast.error(error.message)
        }
    }


    //fucntion to get messages for selected user
    const getMessages = async (userId)=>{
        try {
           const { data } = await axios.get(`/api/messages/${userId}`);
           if (data.success){
            setMessages(data.messages)
           }
        } catch (error) {
            toast.error(error.message)
        }
    }

    //function to send message to selected user
    

    const value = {

    }

    return( 
    <ChatContext.Provider value={value}>
        { children }
    </ChatContext.Provider>
    )
}