from dotenv import load_dotenv
from .states import GenerateAnalystsState
from .models import llm
from .objects import Analyst, Perspective
from .prompts import analyst_instructions
from langchain.messages import SystemMessage, HumanMessage
from langgraph.types import interrupt

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
