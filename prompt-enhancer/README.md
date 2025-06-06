# Advanced Prompt Enhancer Pro 🚀

A state-of-the-art prompt enhancement tool that implements advanced prompting techniques from both Anthropic and OpenAI to optimize your prompts for maximum effectiveness.

## Features

### 🔷 Anthropic Techniques
- **XML Structure**: Clear task organization with XML tags
- **Thinking Tags**: Allow the model to reason before responding
- **Answer Leveling**: Control response depth (1-10 scale)
- **Prefill**: Start the assistant's response to set tone
- **Human/Assistant Tags**: Clear conversation structure

### 🔶 OpenAI Techniques
- **System Messages**: Set behavior and expertise level
- **Delimiters**: Clear input boundaries (triple quotes, XML, markdown)
- **Few-shot Learning**: Provide examples for pattern recognition
- **Step-by-step Instructions**: Detailed process breakdown
- **Temperature Guidance**: Balance creativity and accuracy

### 🔸 Advanced Features
- **Variable Substitution**: Use `{{variable_name}}` syntax for dynamic content
- **File Upload Support**: Upload PDFs or text files for variable content
- **Meta-prompting**: Have AI plan its approach before responding
- **Self-consistency Checks**: Verify response quality
- **Error Handling**: Gracefully handle edge cases
- **Task Decomposition**: Break complex requests into manageable steps
- **Smart Length Trimming**: Preserve structure when limiting prompt length

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd prompt-enhancer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app-v2.py
```

2. Open your browser to `http://localhost:8501`

3. Enter your prompt in the left panel

4. Configure enhancement options in the sidebar:
   - Select task type (general, code generation, analysis, etc.)
   - Choose which techniques to apply
   - Set parameters like answer depth level
   - Enable/disable specific enhancements

5. Click "Enhance Prompt" to generate the optimized version

6. Use variables with `{{variable_name}}` syntax:
   - Variables are automatically detected
   - Upload files or enter text for each variable
   - Test the prompt with Claude API with variables substituted

## Example Enhancements

### Data Analysis with Variables
**Original**: `analyze {{sales_data}} and find trends`

**Enhanced**: Includes system message, XML structure, chain of thought, output format specification, error handling, and self-consistency checks.

### Code Generation with Answer Leveling
**Original**: `create a function to process user input`

**Enhanced**: Features thinking tags, step-by-step implementation plan, few-shot examples, security constraints, and detailed implementation at level 7/10.

### Creative Writing with Meta-prompting
**Original**: `write a story about {{character}} in {{setting}}`

**Enhanced**: Incorporates meta-prompt for planning, novelist role, temperature guidance, structured output, character development constraints, and narrative prefill.

## Configuration Options

### Enhancement Techniques
- **XML Structure**: Organize prompts with clear XML tags
- **Thinking Tags**: Add reasoning process before response
- **Answer Leveling**: Control response depth (1-10)
- **System Message**: Define AI behavior and expertise
- **Delimiters**: Choose input boundary style
- **Chain of Thought**: Step-by-step reasoning
- **Few-shot Examples**: Learn from examples
- **Constraints**: Quality guidelines
- **Error Handling**: Handle edge cases
- **Meta-prompting**: Self-reflection
- **Self-consistency**: Verify responses

### Advanced Options
- **Variables**: Enable `{{variable}}` syntax
- **Temperature Guidance**: Control creativity (0.0-2.0)
- **Max Length**: Limit prompt length
- **Test with API**: Validate enhanced prompts

## API Configuration

The app uses the Claude API through LiteLLM. Update these settings in the code:
```python
API_BASE_URL = "https://litellm.deriv.ai/v1"
OPENAI_API_KEY = "your-api-key"
OPENAI_MODEL_NAME = "claude-4-sonnet"
```

## Best Practices

1. **Be Specific**: Provide detailed requirements
2. **Use Examples**: Enable few-shot learning for complex patterns
3. **Define Output Format**: Specify exactly what you want
4. **Add Context**: Background information improves results
5. **Test Iteratively**: Use API testing to refine prompts
6. **Leverage Variables**: Use for dynamic content insertion

## Requirements

- Python 3.8+
- Streamlit
- Requests
- PyPDF2

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 