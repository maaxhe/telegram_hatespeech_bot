README.md

# Telegram Hate Speech Detection Bot

## Overview

This Telegram bot identifies hate speech in messages and responds accordingly. It uses a pre-trained `RoBERTa` model from Hugging Face for hate speech detection and integrates with OpenAI's GPT-3.5-turbo for generating responses. The bot tracks hate speech occurrences and can warn, delete messages, and kick users after repeated offenses.

## Features

- **Hate Speech Detection**: Detects hate speech using the `facebook/roberta-hate-speech-dynabench-r4-target` model.
- **GPT-3.5 Response Generation**: Responds to hate speech with a creative warning message.
- **User Tracking**: Tracks users who send hate speech, warns them, and kicks them if they persist.
- **Custom Commands**:
  - `/start`: Welcome message.
  - `/help`: Instructions for using the bot.
  - `/hatecount`: Displays the number of hate messages sent by the user.
  
## Setup

### Requirements

- Python 3.7+
- Telegram Bot API token
- OpenAI API key
- Hugging Face Transformers model

### Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Replace the following placeholders in the code:
   - `openai.api_key`: Your OpenAI API key.
   - `ApplicationBuilder().token(...)`: Your Telegram Bot API token.

4. Run the bot:

   ```bash
   python bot.py
   ```

## Usage

1. Start the bot on Telegram with the `/start` command.
2. Send a message, and the bot will check if it contains hate speech.
3. If hate speech is detected:
   - The bot will respond with a warning message.
   - After 3 hate messages, the bot will issue a final warning.
   - After 4 hate messages, the user will be kicked from the chat.
4. Use `/hatecount` to see how many hate messages you've sent.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

### requirements.txt

```
python-telegram-bot>=20.0a0
openai>=0.27.0
transformers>=4.30.2
torch>=2.0.0
httpx>=0.23.0
```
