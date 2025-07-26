# Lesson 17: Introduction to Multi-Agent Systems with AutoGen

This lesson introduces the fundamentals of building multi-agent systems using the AutoGen framework. You will learn how to orchestrate multiple AI agents to collaborate on complex tasks, leveraging their specialized capabilities to achieve a common goal.

## Key Learning Objectives

- **Multi-Agent Concepts**: Understand the core principles of multi-agent systems, including collaboration, communication, and role-based specialization.
- **AutoGen Framework**: Learn the key components of AutoGen, such as `ConversableAgent` and `UserProxyAgent`.
- **Agent Orchestration**: See how to define a group of agents and orchestrate their interaction to solve a problem.
- **Reflexion Mechanism**: Implement LLM reflection using nested chats for self-critique and improvement.
- **Practical Applications**: Explore common use cases for multi-agent systems, such as automated problem-solving, code generation, and content creation.

## File Descriptions

- `README.md`: This file, providing an overview of the lesson.
- `requirements.txt`: Lists all the necessary Python dependencies for this lesson.
- `reflexion_agent_demo.py`: A complete demonstration of Reflexion agent implementation using AutoGen.
- `simple_reflexion_example.py`: A simplified version showing core concepts.

## Reflexion Agent Demo

The `reflexion_agent_demo.py` file demonstrates how to implement a Reflexion agent using AutoGen's nested chat functionality. This implementation is based on the [Microsoft AutoGen documentation](https://microsoft.github.io/autogen/0.2/docs/topics/prompting-and-reasoning/reflection/).

### Key Features

1. **Multi-Agent Architecture**: 
   - `UserProxyAgent`: Manages the conversation flow and user interaction
   - `WritingAssistant`: Generates content based on user requests
   - `ReflectionAssistant`: Provides critique and improvement suggestions

2. **Reflexion Mechanism**: 
   - Uses nested chats to implement self-reflection
   - Automatically triggers reflection after content generation
   - Provides detailed feedback for content improvement

3. **Practical Examples**:
   - AI technology blog post writing
   - Sustainable development content creation
   - Demonstrates iterative improvement through reflection

4. **LLM Configuration**:
   - Uses DashScope API with Qwen models
   - Compatible with the main project's LLM configuration
   - Supports environment variable configuration

### How It Works

1. **Initial Request**: User provides a writing task to the system
2. **Content Generation**: Writing assistant creates initial content
3. **Automatic Reflection**: Reflection assistant analyzes the content and provides critique
4. **Iterative Improvement**: Writing assistant revises content based on feedback
5. **Final Output**: Improved content is delivered to the user

## Setup and Execution

1. **Install Dependencies**:
   Install the required packages using pip.
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   Set your DashScope API key as an environment variable:
   ```bash
   export DASHSCOPE_API_KEY='your-api-key-here'
   export LLM_MODEL='qwen-max'  # Optional, defaults to qwen-max
   ```

3. **Run the Reflexion Demo**:
   Execute the demonstration script:
   ```bash
   python reflexion_agent_demo.py
   ```

4. **Run the Simple Example**:
   For a quick test of core concepts:
   ```bash
   python simple_reflexion_example.py
   ```

## Expected Output

The demo will show:
- Initial content generation by the writing assistant
- Automatic reflection and critique by the reflection assistant
- Improved content based on the feedback
- Interactive prompts to continue between demonstrations

## Learning Outcomes

After completing this lesson, you will understand:
- How to create and configure AutoGen agents
- How to implement nested chat functionality for reflection
- How to orchestrate multiple agents for collaborative tasks
- How to use caching to improve performance
- Best practices for multi-agent system design
- How to integrate with DashScope API and Qwen models

## Next Steps

This lesson provides a foundation for building more complex multi-agent systems. Consider exploring:
- Advanced agent communication patterns
- Integration with external tools and APIs
- Custom agent specialization for specific domains
- Performance optimization and scaling strategies 