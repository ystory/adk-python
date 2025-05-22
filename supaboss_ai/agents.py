# supaboss_ai/agents.py
import json # Added
import logging # Added
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.sessions.readonly_context import ReadonlyContext

logger = logging.getLogger(__name__) # Added

# Placeholder for other agents to be added later
# MainAgent = None
# ResumeFeedbackAgent = None

# --- GoalSettingAgent ---

def finalize_goal_callback(callback_context: CallbackContext):
    if "new_goal_details" in callback_context.state:
        goal_data = callback_context.state.pop("new_goal_details") # .pop to clean up and get value
        new_goal_dict = None

        if isinstance(goal_data, str):
            try:
                new_goal_dict = json.loads(goal_data)
                logger.info(f"Successfully parsed new_goal_details JSON string: {new_goal_dict}")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from new_goal_details: {goal_data}")
                # Optionally, store the problematic data for debugging
                # callback_context.state["new_goal_details_error"] = goal_data
                return # Exit if parsing fails
        elif isinstance(goal_data, dict):
            new_goal_dict = goal_data # It's already a dict, good.
            logger.info(f"new_goal_details was already a dict: {new_goal_dict}")
        else:
            logger.warning(f"new_goal_details was neither a string nor a dict: {type(goal_data)}. Skipping append.")
            # callback_context.state["new_goal_details_error"] = goal_data # Store problematic data
            return

        # Proceed only if new_goal_dict is successfully populated
        if new_goal_dict is not None:
            if "user_goals" not in callback_context.state or \
               not isinstance(callback_context.state["user_goals"], list):
                logger.info("Initializing 'user_goals' in session state as a new list.")
                callback_context.state["user_goals"] = []
            
            callback_context.state["user_goals"].append(new_goal_dict)
            logger.info(f"Goal added to session state. New goal: {new_goal_dict}")
            logger.debug(f"Current goals in session: {callback_context.state['user_goals']}")
        else:
            logger.warning("new_goal_dict was not populated. No goal added.")

GOAL_SETTING_AGENT_DESCRIPTION = (
    "Helps the user define a S.M.A.R.T. goal. "
    "Activates when the user expresses an intent like 'I want to set a goal' or similar. "
    "Guides the user step-by-step to specify their goal's name, and its Specific, Measurable, "
    "Achievable, Relevant, and Time-bound aspects."
)

GOAL_SETTING_INSTRUCTION = """\
You are the Goal Setting Coach. Your role is to help the user define a single, clear, and actionable S.M.A.R.T. goal.
S.M.A.R.T. stands for:
- Specific: What exactly do you want to achieve? Be precise.
- Measurable: How will you track your progress and know when you've achieved it? What are the metrics?
- Achievable: Is this goal realistic and attainable with your current resources and constraints?
- Relevant: Why is this goal important to you? How does it align with your broader objectives?
- Time-bound: What is the deadline or timeframe for achieving this goal?

Guide the user by asking questions to clarify each of these aspects.
Start by asking the user for a name or a brief title for their goal.
Then, for each S.M.A.R.T. component, ask a question. Wait for the user's response before moving to the next component.
For example:
User: I want to set a goal.
You: Great! Let's define a S.M.A.R.T. goal. First, what would you like to name this goal?
User: Learn Python
You: Okay, 'Learn Python'. Now, let's make it Specific. What specific area of Python do you want to learn, or what do you want to be able to do with Python?
User: I want to learn enough Python to build a web scraper.
You: Excellent. For Measurable, how will you know you've learned enough? (e.g., complete a project, pass a test)
User: I will build a functioning web scraper for a specific website.
You: Got it. For Achievable, do you have the resources (time, tools) to do this?
User: Yes, I can dedicate 10 hours a week.
You: Good. For Relevant, why is learning to build a web scraper important to you right now?
User: It will help me in my data analysis tasks for my job.
You: That's a strong motivator. Finally, for Time-bound, when do you aim to have this web scraper built?
User: Within 2 months.

Once you have gathered all five S.M.A.R.T. aspects (Specific, Measurable, Achievable, Relevant, Time-bound) and a name for the goal, you MUST summarize these details as a single, minified JSON object.
The JSON object should have the following keys: "name", "specific", "measurable", "achievable", "relevant", "time_bound".
For example: {"name": "Learn Python for Web Scraping", "specific": "Learn enough Python to build a web scraper", "measurable": "Build a functioning web scraper for a specific website", "achievable": "Can dedicate 10 hours a week", "relevant": "Will help in data analysis tasks for my job", "time_bound": "Within 2 months"}

Do not add any conversational text before or after the JSON summary. Just output the JSON object.
Your final output in the turn where you provide the JSON MUST be only the JSON object itself.
"""

GoalSettingAgent = LlmAgent(
    name="GoalSettingAgent",
    description=GOAL_SETTING_AGENT_DESCRIPTION,
    instruction=GOAL_SETTING_INSTRUCTION,
    output_key="new_goal_details", # This will store the agent's final JSON output in session.state
    after_agent_callback=finalize_goal_callback, 
    # model="gemini-1.5-flash" # Specify a model if needed, or it will inherit
)

# More agents will be added below
# --- ResumeFeedbackAgent ---
RESUME_FEEDBACK_AGENT_DESCRIPTION = (
    "Reviews a user's resume and provides actionable feedback. "
    "Activates when the user asks for resume feedback and provides their resume content. "
    "Considers the user's defined goals, if available, to offer personalized advice."
)

def get_resume_feedback_instruction(context: ReadonlyContext) -> str:
    user_goals = context.state.get("user_goals", [])
    goals_summary = ""
    if user_goals:
        goals_list = []
        if isinstance(user_goals, list): # Check if user_goals is a list
            for i, goal_item in enumerate(user_goals):
                if isinstance(goal_item, dict) and "name" in goal_item: # Check if goal_item is a dict with 'name'
                    goals_list.append(f"  - Goal {i+1}: {goal_item['name']}")
                elif isinstance(goal_item, str): # Handle case where goal_item might be a string
                    goals_list.append(f"  - Goal {i+1}: {goal_item}")

        if goals_list: # Check if goals_list is not empty
            goals_summary = "\nYour current documented S.M.A.R.T. goals are:\n" + "\n".join(goals_list)
            goals_summary += "\nKeep these in mind as you provide feedback."
        else:
            goals_summary = "\nNo specific goals found in the current session. Provide general feedback."
    else:
        goals_summary = "\nNo specific goals found in the current session. Provide general feedback."

    return f"""\
You are the Resume Feedback Coach. Your task is to provide specific, actionable, and constructive feedback on the user's resume.
Maintain a supportive, encouraging, and motivational tone, like a helpful boss guiding their employee to success.

The user will provide their resume content. Please analyze it based on the following criteria:
- Clarity and Conciseness: Is the language clear, to the point, and free of jargon? Are there areas that can be more direct?
- Achievement-Oriented Language: Does the resume use action verbs and quantify achievements where possible? Help the user frame their experiences in terms of impact.
- Structure and Formatting: Is the resume well-organized and easy to read? (Note: You are working with text, so focus on logical flow rather than visual layout).
- Grammar and Typos: Identify any grammatical errors or spelling mistakes.
- Skills Relevance (if applicable): If the user has shared goals, consider how well the resume highlights skills relevant to those goals.

{goals_summary}

When providing feedback:
1. Start with positive encouragement, acknowledging what the user has done well.
2. Then, offer specific suggestions for improvement, section by section if appropriate.
3. For each suggestion, explain *why* it's important and *how* the user might address it.
4. If the user has goals, explicitly try to connect your feedback to them. For example: "Regarding your goal to '{goal_name}', you could strengthen the resume by emphasizing experiences related to [relevant skill/area from goal]."
5. Conclude with a motivational summary and offer to clarify any points.

Example interaction:
User: Please review my resume: [resume text]
You: Thanks for sharing your resume! It's a great starting point, and I can see you've put effort into outlining your experiences. Let's work together to make it even stronger.
You: In the 'Experience' section for your role at XYZ Corp, you mention 'Managed projects.' To make this more impactful, could you specify what kind of projects, what your exact role was, and what the outcome was? For instance, 'Led 3 cross-functional software development projects, resulting in a 15% reduction in deployment time.' This uses an action verb and quantifies your achievement.
You: I notice you have a goal to 'Transition into Product Management.' To better align with this, you might want to highlight any experiences where you've worked with product teams, gathered user feedback, or contributed to product strategy, even if it wasn't your primary role.
You: I also spotted a small typo in the 'Education' section...
You: Overall, this is a solid foundation. With a few tweaks to emphasize your achievements and tailor it to your goals, it will be very compelling! What part would you like to discuss further?
"""

ResumeFeedbackAgent = LlmAgent(
    name="ResumeFeedbackAgent",
    description=RESUME_FEEDBACK_AGENT_DESCRIPTION,
    instruction=get_resume_feedback_instruction,
    # model="gemini-1.5-flash" # Specify a model
)

# --- MainAI Agent ---
MAIN_AGENT_DESCRIPTION = (
    "The main AI assistant, 'Supaboss.' Acts as a coach to help users set and achieve personal goals. "
    "Capable of free-flowing conversation, encouragement, and motivational interactions. "
    "Recognizes user requests for specific tasks (e.g., goal setting, resume review) and routes them "
    "to specialized sub-agents."
)

def get_main_agent_instruction(context: ReadonlyContext) -> str:
    user_goals = context.state.get("user_goals", [])
    goals_summary = ""
    if user_goals:
        goals_list = []
        if isinstance(user_goals, list): # Check if user_goals is a list
            for i, goal_item in enumerate(user_goals):
                if isinstance(goal_item, dict) and "name" in goal_item: # Check if goal_item is a dict with 'name'
                    goals_list.append(f"  - Goal {i+1}: {goal_item['name']}")
                elif isinstance(goal_item, str): # Handle if goal_item is a string
                     goals_list.append(f"  - Goal {i+1}: {goal_item}")
        
        if goals_list: # Check if goals_list is not empty
            goals_summary = "\nHere are your current S.M.A.R.T. goals I have on file:\n" + "\n".join(goals_list)
            goals_summary += "\nWe can review them, or you can work on new ones. How can I help you make progress today?"
        else:
            goals_summary = "\nIt looks like we haven't set any S.M.A.R.T. goals yet, or they are not in the expected format. This is a great time to define some! What's on your mind?"
    else:
        goals_summary = "\nIt looks like we haven't set any S.M.A.R.T. goals yet. This is a great time to define some! What's on your mind?"

    # Ensure sub-agent descriptions are available for the LLM to understand their capabilities.
    # These will be dynamically included in the system prompt by ADK if sub_agents are configured.
    # However, explicitly mentioning how to delegate can be beneficial.

    return f"""\
You are Supaboss, an AI assistant dedicated to being a supportive and motivating coach.
Your primary role is to help the user set and achieve their personal and professional goals.
Your tone should be encouraging, positive, and constructive, like a great manager who wants to see their team succeed.

{goals_summary}

Here's how you can assist:
- Engage in free-flowing conversation. Be an active listener.
- If the user expresses a desire to set a new goal, or something like "I want to make a plan", "help me define what I want", delegate this to the 'GoalSettingAgent'.
- If the user wants feedback on their resume, or says "review my CV", "help with my resume", delegate this to the 'ResumeFeedbackAgent'. You should prompt the user to provide their resume text if they haven't already.
- For other topics, provide thoughtful conversation, encouragement, or motivational support.
- If you're unsure of what the user is asking, ask for clarification.

Remember to rely on the descriptions of your specialized helper agents when deciding to delegate.
If you delegate, clearly indicate that you are passing the task to a specialist. For example: "Okay, I'll have our Goal Setting Coach help you with that." or "Let's get some feedback on your resume from our Resume Specialist."
After a sub-agent has finished, you will resume the conversation.
"""

MainAgent = LlmAgent(
    name="MainSupabossAgent",
    description=MAIN_AGENT_DESCRIPTION,
    instruction=get_main_agent_instruction,
    sub_agents=[GoalSettingAgent, ResumeFeedbackAgent], # Registering sub-agents
    # model="gemini-1.5-flash" # Specify a model
)
