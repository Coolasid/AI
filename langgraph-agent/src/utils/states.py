from typing_extensions import TypedDict, NotRequired, Annotated
from typing import Optional, List
from .objects import Analyst
from langgraph.graph import MessagesState
import operator

class GenerateAnalystsState(TypedDict):
    topic: str # Research Topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: NotRequired[Optional[str]] # Human feedback for what is generated
    analysts: NotRequired[List[Analyst]] # List of all our Analysts
    

class InteriewState(MessagesState):
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, operator.add] # source of docs
    analyst: Analyst # my analyst
    interview: str # interview transcript
    sections: list # final key we duplicate in outer stae for Send() api