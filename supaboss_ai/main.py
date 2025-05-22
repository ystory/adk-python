# supaboss_ai/main.py
import asyncio
import logging

from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part # Will uncomment when chat loop is added

from .agents import MainAgent

# Configure basic logging
logging.basicConfig(level=logging.INFO) # Use DEBUG for more verbose ADK logs
logger = logging.getLogger(__name__)

async def run_cli_chat():
    print("Supaboss AI CLI - Initializing...")

    app_name = "supaboss_ai_cli"
    user_id = "cli_user_01" # Fixed user ID for this simple CLI
    session_id = "supaboss_cli_session_01" # Fixed session ID for continuity

    runner = InMemoryRunner(
        app_name=app_name,
        agent=MainAgent, # MainAgent should be the root agent from agents.py
    )

    # Get or create a session with a fixed ID
    session = await runner.session_service.get_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    if not session:
        logger.info(f"Creating new session: {session_id}")
        session = await runner.session_service.create_session(
            app_name=app_name, user_id=user_id, session_id=session_id
        )
    else:
        logger.info(f"Resumed session: {session_id}")

    if not session:
        logger.error("Failed to create or retrieve a session. Exiting program.")
        return

    print(f"Session '{session.id}' is active.")

    # NEW WELCOME MESSAGES AND CHAT LOOP:
    print("\nWelcome to Supaboss AI!") 
    print("I'm here to help you as your AI coach. How can I assist you today?")
    print("Type 'quit' or 'exit' to end the session.")
    print("----------------------------------------------------")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Supaboss AI: It was great working with you. Goodbye!")
                break
            if not user_input.strip():
                continue
            
            message = Content(role="user", parts=[Part.from_text(user_input)])
            
            current_response_text = ""
            print("Supaboss AI: ", end="", flush=True)

            async for event in runner.run_async(
                user_id=user_id, # user_id should be defined in the outer scope
                session_id=session.id, # session should be defined in the outer scope
                new_message=message,
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            print(part.text, end="", flush=True)
                            current_response_text += part.text
                        elif part.function_call:
                            logger.debug(f"Agent requested function call: {part.function_call.name} with args {part.function_call.args}")
            
            print() # Newline after agent's full response
            if not current_response_text.strip():
                 pass # Placeholder for if agent only makes tool calls

        except KeyboardInterrupt:
            print("\nSupaboss AI: Session interrupted. Goodbye!")
            break
        except Exception as e:
            logger.error(f"An error occurred during the chat loop: {e}", exc_info=True)
            print("Supaboss AI: I encountered an issue. Please try again or restart the session.")

if __name__ == "__main__":
    try:
        asyncio.run(run_cli_chat())
    except KeyboardInterrupt:
        print("\nExiting CLI.")
    except Exception as e:
        logger.error(f"CLI terminated with an error: {e}", exc_info=True)
