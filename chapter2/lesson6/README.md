# Lesson 6: Managing Conversation History

This lesson focuses on a critical aspect of building effective chatbots: managing conversation history. You will learn techniques for maintaining context and summarizing dialogues to ensure coherent and efficient interactions.

## Key Learning Objectives

- **History Management**: Understand how to maintain and pass conversation history to an LLM to provide context for its responses.
- **Context Window Management**: Learn a basic technique for controlling the size of the conversation history to fit within a model's context window.
- **Dialogue Summarization**: Explore how to summarize long conversations to retain key information while reducing the amount of text sent to the model.

## File Descriptions

- `example1.py`: Demonstrates a customer service chatbot that maintains a rolling window of the conversation history to provide context for its replies.
- `example2.py`: Shows how to generate a summary of a lengthy dialogue, extracting key information to be used as context in future turns.
- `requirements.txt`: Lists all the necessary Python dependencies for this lesson.

## Setup and Execution

1.  **Install Dependencies**:
    Install the required packages using pip.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Examples**:
    Execute the individual scripts to see the history management techniques.
    ```bash
    # For rolling history management
    python example1.py

    # For dialogue summarization
    python example2.py
    ``` 