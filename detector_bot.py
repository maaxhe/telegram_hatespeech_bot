#!/usr/bin/env python
# pylint: disable=unused-argument

import logging
import openai
import time
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, ApplicationBuilder
from transformers.pipelines import pipeline as torch_pipeline

# Model configuration
model_name = "facebook/roberta-hate-speech-dynabench-r4-target"

# OpenAI API key (replace with your own)
openai.api_key = "YOUR OPEN-AI-KEY"

# Track user hate messages count
user_hate_message_count = {}

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# Set a higher logging level for httpx to avoid excessive logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Get logger instance
logger = logging.getLogger(__name__)

# --- Bot Command Handlers ---

# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /start command: Sends a welcome message to the user."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}!",  # Sends a personalized greeting
        reply_markup=ForceReply(selective=True),  # Force reply option
    )

# Help command handler
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /help command: Sends help instructions."""
    await update.message.reply_text("Help!")

# --- Hate Speech Detection and Processing ---

# Decorator to measure the execution time of a function
def time_func(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Execution time: {time.time() - start}")
        return result
    return wrapper

# Load hate speech detection model
torch_pipe = torch_pipeline("text-classification", model_name, device="mps")

@time_func
def pipe(text):
    """Processes the input text through the hate speech detection model."""
    print(text)
    result = torch_pipe(text)
    print(result)
    return result

# Query OpenAI GPT model with user prompt
async def query_llm(prompt: str) -> str:
    """Queries the GPT model with a prompt and returns the response."""
    try:
        # Call the OpenAI API using the ChatCompletion endpoint
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a conversational bot. If you get insulted, insult back creatively."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,  # Limit response length
            temperature=0.7,  # Controls creativity
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Query OpenAI GPT model for hate speech response
async def hate_query_llm(prompt: str) -> str:
    """Queries GPT with a creative response to hate speech."""
    new_prompt = f"Someone said this in a groupchat: {prompt}. Please answer creatively and remind them that hate speech will not be tolerated."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a conversational bot."},
                {"role": "user", "content": new_prompt},
            ],
            max_tokens=150,
            temperature=0.7,
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Hate message count handler
async def hate_message_count(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends the count of hate messages a user has sent."""
    user_id = update.message.from_user.id
    user_name = update.message.from_user.username or update.message.from_user.first_name

    count = user_hate_message_count.get(user_id, 0)  # Get hate message count for user
    await update.message.reply_text(f"{user_name}, you have sent {count} hate messages.")

# --- Message Handler ---

# Handle incoming messages and check for hate speech
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Processes incoming messages and checks for hate speech."""
    user_input = update.message.text
    user_id = update.message.from_user.id
    user_name = update.message.from_user.username or update.message.from_user.first_name

    # If message starts with 'bot', strip the prefix for processing
    if user_input[:4] == 'bot ':
        mod_user_input = user_input[4:]
    else:
        mod_user_input = user_input 
    # Detect hate speech
    hate_speech_detection = pipe(mod_user_input)
    hate_label = hate_speech_detection[0]['label']

    if hate_label == "hate":
        # Increment hate message count for the user
        user_hate_message_count[user_id] = user_hate_message_count.get(user_id, 0) + 1
        logger.info(f"{user_name} has sent {user_hate_message_count[user_id]} hate messages.")

        # Respond to hate speech using GPT
        response = await query_llm(f"This message contains hate speech: {user_input}")
        await update.message.reply_text(response)
        await update.message.reply_text("✨")

        # Warn or ban user after multiple hate messages
        if user_hate_message_count[user_id] == 3:
            await update.message.reply_text("‼️ You will be banned for sending hate speech if it happens again. 😡")
        if user_hate_message_count[user_id] == 4:
            try:
                # Ban the user from the chat
                await context.bot.ban_chat_member(update.effective_chat.id, user_id)
                await context.bot.unban_chat_member(update.effective_chat.id, user_id)
                await update.message.reply_text(f"User {user_name} has been kicked from the group.")
                user_hate_message_count[user_id] = 0  # Reset hate message count
            except Exception as e:
                logger.error(f"Failed to kick user: {e}")
                await update.message.reply_text(f"Failed to kick user {user_id}. Error: {e}")

        # Delete the hate message
        try:
            await update.message.delete()
            logger.info(f"Deleted a hate message from user {user_name}")
            await update.message.reply_text(f"{user_name}, your message was deleted due to prohibited content.")
        except Exception as e:
            logger.error(f"Failed to delete message: {e}")
    else:
        if user_input[:4] == 'bot ':
            # Normal message handling (not hate speech)
            response = await query_llm(user_input[4:])
            await update.message.reply_text(response)

# --- Main Function ---

def main() -> None:
    """Start the bot and configure the command/message handlers."""
    # Create the Application and pass in the bot's token
    app = ApplicationBuilder().token("YOUR TELEGRAM-TOKEN").build()

    # Command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("hatecount", hate_message_count))

    # Message handler for all text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot until manually stopped
    app.run_polling(allowed_updates=Update.ALL_TYPES)

# Run the bot
if __name__ == "__main__":
    main()
