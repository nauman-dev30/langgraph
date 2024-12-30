from fastapi import FastAPI
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langgraph.graph import MessagesState
import os
import time

load_dotenv()

# Initialize LLM with proper parameters
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,  # Add temperature for more consistent responses
    max_output_tokens=2048  # Set maximum output length
)

loader = TextLoader("data.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever(
    search_kwargs={"k": 3}  # Specify number of results to retrieve
)

info_retriever = create_retriever_tool(
    retriever,
    "hotel_information_sender",
    "Searches information about hotel from provided vector and return accurate as you can",
)

tools = [info_retriever]

llm_with_tools = llm.bind_tools(tools)

sys_msg = (
    "You are Alexandra Hotel's virtual assistant, trained to assist customers with any queries related to the hotel. "
    "Your primary responsibility is to provide accurate, helpful, and friendly responses. "
    "You have access to a specialized tool for retrieving detailed and up-to-date information about the hotel, "
    "such as amenities, room availability, pricing, dining options, events, and policies. Use this tool effectively to provide precise answers. "
    "If a query is beyond your scope or requires external actions (e.g., booking confirmation, cancellations), "
    "politely inform the user and guide them to contact the hotel's staff for further assistance. "
    "Maintain a professional yet approachable tone at all times."
    "Must use the info_retriever tool to access hotel related information."
)

def assistant(state: MessagesState):
    try:
        response = llm_with_tools.invoke([sys_msg] + state["messages"][-10:])
        return {"messages": [response]}
    except Exception as e:
        print(f"Error in assistant: {str(e)}")
        # Return a default response if there's an error
        return {"messages": [{"role": "assistant", "content": "I apologize, but I encountered an issue. Could you please try again?"}]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory = MemorySaver()

agent = builder.compile(checkpointer=memory)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/chat/{query}")
def get_content(query: str):
    print(f"Query received: {query}")
    try:
        # Add retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use timestamp for unique thread_id
                config = {"configurable": {"thread_id": str(time.time())}}
                result = agent.invoke({"messages": [("user", query)]}, config)
                print(f"Agent result: {result}")
                return {"messages": result["messages"]}
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(1)  # Wait before retry
                
    except Exception as e:
        print(f"Error: {e}")
        return {"messages": [{"role": "assistant", "content": "Sorry, an error occurred. Please try again."}]}