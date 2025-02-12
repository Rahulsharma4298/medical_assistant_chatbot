from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from tools import search_medicine, medical_search

load_dotenv()

tools = [search_medicine, medical_search]
model = ChatGoogleGenerativeAI(model='gemini-1.5-flash').bind_tools(tools)
ROOT_NODE = "MedicalAssistant"


class State(MessagesState):
    pass

def medical_assistant(state: State):
    system_prompt = """You are a helpful medical assistant.
    Help with user for their medical related queries.
    You will get context from 'medicine_search', use it to generate response.
    Keep the answer detailed but concise. 
    Format the response as Markdown bullet points.
    Also help them with the medicines they are looking for.
    Question: """
    question = state['messages']
    print(question)
    resp = model.invoke([SystemMessage(system_prompt)] + question)
    return {"messages": resp}



builder = StateGraph(State)
builder.add_node(ROOT_NODE, medical_assistant)
builder.add_node('tools', ToolNode(tools))
builder.add_edge(START, ROOT_NODE)
builder.add_conditional_edges(ROOT_NODE, tools_condition)
builder.add_edge('tools', ROOT_NODE)
builder.add_edge(ROOT_NODE, END)
graph = builder.compile()

def chat(query):
    return graph.invoke({'messages': query}, output_keys='messages')[-1].content