# Sales Agent Application Documentation

## Overview

This application is a Streamlit-based chat interface that simulates a conversation with a sales agent. It uses the LangChain library to generate responses based on the conversation history and current stage of the sales process.

## Components

1. **Configuration Files**:
   - `config.json`: Contains general settings for the sales agent (name, role, company details, etc.).
   - `conversation_stages.json`: Defines the different stages of a sales conversation.

2. **Main Script** (`app.py`):
   - Implements the Streamlit interface and the core logic of the sales agent.

3. **External Libraries**:
   - Streamlit: For creating the web interface.
   - LangChain: For natural language processing and response generation.
   - OpenAI: As the underlying language model (accessed through LangChain).

## Key Functions

### `determine_conversation_stage(conversation_history)`

- **Purpose**: Analyzes the conversation history to determine the current stage of the sales conversation.
- **Input**: List of conversation messages.
- **Output**: A string describing the current conversation stage.
- **Process**: Uses the `stage_analyzer_chain` to predict the current stage based on the conversation history.

### `generate_response(conversation_history, current_stage)`

- **Purpose**: Generates the sales agent's response based on the conversation history and current stage.
- **Inputs**: 
  - `conversation_history`: List of previous messages.
  - `current_stage`: Current stage of the conversation.
- **Output**: A string containing the agent's response.
- **Process**: Uses the `sales_conversation_chain` to generate an appropriate response.

### `main()`

- **Purpose**: Main function that sets up the Streamlit interface and manages the conversation flow.
- **Process**:
  1. Initializes the conversation with a greeting from the sales agent.
  2. Displays the conversation history.
  3. Accepts user input.
  4. Determines the conversation stage.
  5. Generates and displays the agent's response.
  6. Updates the conversation history.

## Data Flow

1. The application loads configuration and conversation stages from JSON files.
2. When a user sends a message:
   a. The message is added to the conversation history.
   b. The `determine_conversation_stage()` function analyzes the updated history.
   c. The `generate_response()` function creates the agent's reply.
   d. The reply is displayed and added to the conversation history.

## Setup and Running

1. Ensure all required libraries are installed:
   ```
   pip install streamlit langchain openai langchain-community python-dotenv logging
   ```

2. Set up your OpenAI API key as an environment variable or in Streamlit secrets.

3. Place `config.json` and `conversation_stages.json` in the same directory as the script.

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Customization

- Modify `config.json` to change the sales agent's details or company information.
- Adjust `conversation_stages.json` to alter the stages of the sales conversation.
- The prompt templates in the main script can be modified to change the behavior of the stage analyzer or response generator.

## Optimization and impoved response time

- **Model Update**: Changed to "gpt-3.5-turbo," which generally provides faster responses.

- **Parameter Adjustment**: Lowered the temperature from 0.9 to 0.7 to encourage faster and more focused responses.

- **Caching**: Implemented the @lru_cache (Least Recently Used) decorator on key functions like determine_conversation_stage and generate_response. This caches results based on the function's input arguments, reducing redundant API calls and speeding up repeated requests.

- **Purpose of @lru_cache**: It stores up to 128 results, discarding the least recently used ones when the limit is reached. This is particularly useful for caching the output of repeated inputs, avoiding unnecessary recomputation. 

- Implemented the @lru_cache (Least Recently Used as @lru_cache(maxsize=128)) decorator on key functions like determine_conversation_stage and generate_response. This caches results based on the function's input arguments, reducing redundant API calls and speeding up repeated requests.

- **Optimized Conversation History Handling**: The conversation history is now consolidated into a single string (conversation_history_str) before being passed to the functions, reducing the overhead of string operations.

- **Logging**: Kept only essential logging to minimize I/O operations.

- **Prompt optimization**: We can refine our prompts to be more concise and focused, which can lead to faster response generation.

## Limitations and Considerations

- The application relies on the OpenAI API, so ensure you have sufficient credits and comply with OpenAI's usage policies.
- The quality of responses depends on the underlying language model and the provided prompts.
- This is a simulated conversation and should not be used as a replacement for real human interaction in critical sales processes.

## Future Improvements

- Implement error handling for API calls and user inputs.
- Add features like sentiment analysis or product recommendation.
- Integrate with a CRM system to log conversations and outcomes.
- Implement a feedback mechanism to improve the agent's responses over time.
- Using Llama or other models with LangChain: Yes, we can use other language models like Llama with LangChain. Here's how we might modify the code to use a different model:

![alt text](llama-1.png)

This documentation provides an overview of the application's structure, key components, and functionality. It can serve as a guide for understanding, using, and potentially extending the sales agent application.