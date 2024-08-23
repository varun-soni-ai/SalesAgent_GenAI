## Variable, Fuctions and Code Flow Explaination

### Variables

1. stage_analyzer_prompt

> Significance: This prompt template is used to guide the language model in determining the current stage of the sales conversation.

> Use: It's used to create the stage_analyzer_chain.

2. stage_analyzer_chain

> Significance: This chain is responsible for analyzing the conversation history and determining the current stage of the sales process.

> Use: It's used in the determine_conversation_stage function.

3. sales_conversation_chain

> Significance: This chain generates the sales agent's responses based on the current conversation stage and history.

> Use: It's used in the generate_response function.

### Functions

1. determine_conversation_stage function

> Significance: This function uses the stage_analyzer_chain to determine the current stage of the conversation.

> Use: It's called in the main function after each user input to update the conversation stage.

2. generate_response function

> Significance: This function uses the sales_conversation_chain to generate the agent's response based on the current stage and conversation history.

> Use: It's called in the main function to generate the agent's response after determining the conversation stage.

### Overall code flow: :

> 1. The application initializes with the configuration and conversation stages loaded from JSON files.

> 2. When a user inputs a message:

>> a. The message is added to the conversation history.

>> b. determine_conversation_stage is called:

>>> It uses stage_analyzer_chain to analyze the conversation history.

>>> It returns the current conversation stage.

>> c. generate_response is called:

>>> It uses sales_conversation_chain to generate a response based on the current stage and history.

>> d. The response is displayed and added to the conversation history.


> 3. This process repeats for each user input, allowing the conversation to progress through different sales stages based on the context.

The significance of this flow is that it allows the sales agent to adapt its responses based on the current stage of the conversation, providing a more natural and effective sales interaction. The stage analysis and response generation are handled by separate components, making the system modular and easier to maintain or extend.

