from dotenv import load_dotenv
from .states import GenerateAnalystsState, InteriewState
from .models import llm
from .objects import Analyst, Perspective, SearchQuery
from .prompts import analyst_instructions, question_instructions, search_instructions
from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt
from langchain_tavily import TavilySearch

load_dotenv()


# nodes
def create_analysts(state: GenerateAnalystsState):
    '''Create analysts'''

    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get('human_analyst_feedback')

    # enforece structred output
    structured_llm = llm.with_structured_output(Perspective)

    # System_message
    system_message = analyst_instructions.format(topic=topic,
                                                 human_analyst_feedback=human_analyst_feedback,
                                                 max_analysts=max_analysts
                                                 )

    # Generate analyst
    analysts = structured_llm.invoke([SystemMessage(system_message)]+[HumanMessage(content='Please Generate the set of analysts.')])

    return {"analysts": analysts}


def human_feedback(state: GenerateAnalystsState):
    '''This is where the human gives feedback about the analyst given'''

    feedback = interrupt({
        "question": "Are these analysts okay for you?",
        "analysts": [
          analysts.model_dump() if hasattr(analysts, 'model_dump') else analysts 
          for analysts in state.get('analysts', [])
        ],
        "instructions": "Return feedback to regenerate analysts or return empty/perfect/continue/okay to approve and continue the graph."

    })

    if feedback is None:
        return {"human_analyst_feedback": None}

    if isinstance(feedback, str):
        feedback = feedback.strip()

        if feedback == '':
            return {"human_analyst_feedback": None}

        if feedback.lower() in {'perfect', 'okay', 'continue', 'yes'}:
            return {'human_analyst_feedback': None}

        return {'human_analyst_feedback': feedback}


def generate_question(state: InteriewState):
    """node to generate the question"""

    #get state analyst
    analyst = state['analyst']

    if isinstance(analyst, dict):
        analyst = Analyst.model_validate(analyst)

    messages = state['messages']

    # generate question 
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)]+messages)

    return {"messages": [question]}


def search_web(state: InteriewState):
    """Retrive docs from the web"""

    # Search query
    structured_llm = llm.with_structured_output(SearchQuery)

    # Search instruction
    system_instruction_system_message = SystemMessage(content=search_instructions)

    tavily_search = TavilySearch(max_result=3)

    search_query = structured_llm.invoke([system_instruction_system_message] + state["messages"])

    # search
    data = tavily_search.invoke({"query": search_query["search_query"]})
    search_docs = data.get("results", data)

    # format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

