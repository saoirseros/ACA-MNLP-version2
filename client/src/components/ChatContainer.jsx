import React, { useContext, useEffect, useRef, useState } from 'react'
import assets, { messagesDummyData } from '../assets/assets'
import { formatMessageTime } from '../lib/utils'
import { ChatContext } from '../../context/ChatContext'
import { AuthContext } from '../../context/AuthContext'
import toast from 'react-hot-toast'

const ChatContainer = () => {

    const { messages, selectedUser, setSelectedUser, sendMessage, 
        getMessages, messageAnalyses, getConversationAnalytics, nlpAvailable} = useContext(ChatContext)

    const { authUser, onlineUsers } = useContext(AuthContext)

    const scrollEnd = useRef()

    const [input, setInput] = useState('');

    // Handle sending a message
    const handleSendMessage = async (e)=>{
        e.preventDefault();
        if(input.trim() === "") return null;
        await sendMessage({text: input.trim()});
        setInput("")
    }

    // Handle sending an image
    const handleSendImage = async (e) =>{
        const file = e.target.files[0];
        if(!file || !file.type.startsWith("image/")){
            toast.error("select an image file")
            return;
        }
        const reader = new FileReader();

        reader.onloadend = async ()=>{
            await sendMessage({image: reader.result})
            e.target.value = ""
        }
        reader.readAsDataURL(file)
    }

    useEffect(()=>{
        if(selectedUser){
            getMessages(selectedUser._id)
            getConversationAnalytics(selectedUser._id)
        }
    },[selectedUser])

    useEffect(()=>{
        if(scrollEnd.current && messages){
            scrollEnd.current.scrollIntoView({ behavior: "smooth"})
        }
    },[messages])

  return selectedUser ? (
    <div className='h-full overflow-scroll relative backdrop-blur-lg'>
      {/* ------- header ------- */}
      <div className='flex items-center gap-3 py-3 mx-4 border-b border-stone-500'>
        <img src={selectedUser.profilePic || assets.avatar_icon} alt="" className="w-8 rounded-full"/>
        <p className='flex-1 text-lg text-white flex items-center gap-2'>
            {selectedUser.fullName}
            {onlineUsers.includes(selectedUser._id) && <span className="w-2 h-2 rounded-full bg-green-500"></span>}
        </p>
        <img onClick={()=> setSelectedUser(null)} src={assets.arrow_icon} alt="" className='md:hidden max-w-7'/>
        <img src={assets.help_icon} alt="" className='max-md:hidden max-w-5'/>
      </div>
      {nlpAvailable === false && (
        <div className='mx-4 mt-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200'>
          NLP analysis is currently unavailable. Messages will still be delivered normally.
        </div>
      )}
      {/* ------- chat area ------- */}
      <div className='flex flex-col h-[calc(100%-120px)] overflow-y-scroll p-3 pb-6'>
        {messages.map((msg, index)=>{
            const analysis = messageAnalyses[msg._id];
            return (
            <div key={index} className={`flex items-end gap-2 justify-end ${msg.senderId !== authUser._id && 'flex-row-reverse'}`}>
                {msg.image ? (
                    <img src={msg.image} alt="" className='max-w-[230px] border border-gray-700 rounded-lg overflow-hidden mb-8'/>
                ):(
                    <div className={`flex flex-col mb-8 max-w-[200px] ${msg.senderId === authUser._id ? 'items-end' : 'items-start'}`}>
                        <p className={`p-2 md:text-sm font-light rounded-lg break-all bg-violet-500/30 text-white ${msg.senderId === authUser._id ? 'rounded-br-none' : 'rounded-bl-none'}`}>{msg.text}</p>
                        {analysis && (
                            <div className='flex flex-wrap gap-1 mt-1 justify-end'>
                                {analysis.sentiment && (
                                    <span className={`text-[10px] px-2 py-0.5 rounded-full ${analysis.sentiment.label === 'positive' ? 'bg-green-600/40 text-green-200' : 'bg-red-600/40 text-red-200'}`}>
                                        {analysis.sentiment.label} ({Math.round(analysis.sentiment.confidence * 100)}%)
                                    </span>
                                )}
                                {analysis.emotion && (
                                    <span className='text-[10px] px-2 py-0.5 rounded-full bg-indigo-600/40 text-indigo-200'>
                                        {analysis.emotion.label}
                                    </span>
                                )}
                                {analysis.toxicity?.label === 'toxic' && (
                                    <span className='text-[10px] px-2 py-0.5 rounded-full bg-red-700/60 text-red-100'>
                                        ⚠ toxic{analysis.toxicity.category ? ` · ${analysis.toxicity.category}` : ''}
                                    </span>
                                )}
                                {analysis.contextLevel && (
                                    <span
                                        className='text-[10px] px-2 py-0.5 rounded-full bg-gray-600/40 text-gray-300'
                                        title={`Adaptive Context Activation: ${analysis.selectedContextMessages ?? 0} prior message(s) used`}
                                    >
                                        ctx: {analysis.contextLevel}{analysis.selectedContextMessages ? ` (${analysis.selectedContextMessages})` : ''}
                                    </span>
                                )}
                            </div>
                        )}
                    </div>
                )}
                <div className="text-center text-xs">
                    <img src={msg.senderId === authUser._id ? authUser?.profilePic || assets.avatar_icon : selectedUser?.profilePic || assets.avatar_icon} alt="" className='w-7 rounded-full' />
                    <p className='text-gray-500'>{formatMessageTime(msg.createdAt)}</p>
                </div>
            </div>
            )
        })}
        <div ref={scrollEnd}></div>
      </div>

{/* ------- bottom area ------- */}
    <div className='absolute bottom-0 left-0 right-0 flex items-center gap-3 p-3'>
        <div className='flex-1 flex items-center bg-gray-100/12 px-3 rounded-full'>
            <input onChange={(e)=> setInput(e.target.value)} value={input} onKeyDown={(e)=> e.key === "Enter" ? handleSendMessage(e) : null} type="text" placeholder="Send a message" 
            className='flex-1 text-sm p-3 border-none rounded-lg outline-none text-white placeholder-gray-400'/>
            <input onChange={handleSendImage} type="file" id='image' accept='image/png, image/jpeg' hidden/>
            <label htmlFor="image">
                <img src={assets.gallery_icon} alt="" className="w-5 mr-2 cursor-pointer"/>
            </label>
        </div>
        <img onClick={handleSendMessage} src={assets.send_button} alt="" className="w-7 cursor-pointer" />
    </div>


    </div>
  ) : (
    <div className='flex flex-col items-center justify-center gap-2 text-gray-500 bg-white/10 max-md:hidden'>
        <img src={assets.logo_icon} className='max-w-16' alt="" />
        <p className='text-lg font-medium text-white'>Chat anytime, anywhere</p>
    </div>
  )
}

export default ChatContainer
