
from dotenv import load_dotenv
from .states import GenerateAnalystsState, InteriewState, ResearchGraphState
from typing import Literal
from langgraph.graph import END
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send



load_dotenv()


# conditional edges
def should_continue(state: GenerateAnalystsState) -> Literal["create_analysts", "__end__"]:
    """Return the next node to excute"""

    human_analyst_feedback = state.get('human_analyst_feedback', None)

    if human_analyst_feedback:
        return 'create_analysts'

    return "__end__"



def routes_messages(state: InteriewState, name: str='expert'):
    '''Route between question and answer'''

    messages = state['messages']
    max_num_turns = state.get("max_num_turns",2)


    #check the number of expert answers
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name== name])

    if num_responses >= max_num_turns:
        return "save_interview"

    return 'ask_question'


def initiate_all_interviews(state: ResearchGraphState):
    """this is the map setup where we run each interview in sub graph using send API"""

    human_analyst_feedback = state.get('human_analyst_feedback')

    if human_analyst_feedback:
        return "create_analysts"
    else:
        topic = state["topic"]
        return [
            Send('conduct_interview', {
                "analyst": analyst, 
                "messages": [HumanMessage(content=f"so you said you were writing an article on {topic}")]
            }) for analyst in state['analysts']
        ]




