from utils.nodes import create_analysts
from langgraph.graph import START, END, StateGraph
from utils.states import GenerateAnalystsState
from dotenv import load_dotenv


load_dotenv()


# creating our graph
# GenerateAnalystsState state
# nodes create+analysts
# edges no need / simple

builder = StateGraph(GenerateAnalystsState)

builder.add_node("create_analysts", create_analysts)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", END)

graph = builder.compile()