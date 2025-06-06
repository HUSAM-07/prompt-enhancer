import streamlit as st
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import requests
import json
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Prompt Enhancer Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "https://litellm.deriv.ai/v1"
OPENAI_API_KEY = "sk-Qwun3BM-Ld3Fo8Lr1I7E5A"
OPENAI_MODEL_NAME = "claude-4-sonnet"

class PromptTechnique(Enum):
    """Enumeration of prompt enhancement techniques"""
    CLARITY = "clarity"
    STRUCTURE = "structure"
    EXAMPLES = "examples"
    CONTEXT = "context"
    CONSTRAINTS = "constraints"
    OUTPUT_FORMAT = "output_format"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    ROLE_PLAY = "role_play"
    TASK_DECOMPOSITION = "task_decomposition"

@dataclass
class EnhancementConfig:
    """Configuration for prompt enhancement"""
    add_xml_tags: bool = True
    add_examples: bool = True
    add_constraints: bool = True
    add_output_format: bool = True
    add_chain_of_thought: bool = False
    add_role: bool = False
    max_length: Optional[int] = None

class PromptEnhancer:
    """Enhances prompts based on Anthropic and OpenAI best practices"""
    
    def __init__(self, config: Optional[EnhancementConfig] = None):
        self.config = config or EnhancementConfig()
        
    def enhance_prompt(self, 
                      original_prompt: str, 
                      task_type: Optional[str] = None,
                      examples: Optional[List[Dict[str, str]]] = None,
                      context: Optional[str] = None,
                      desired_format: Optional[str] = None,
                      role: Optional[str] = None) -> str:
        """
        Enhance a prompt using multiple techniques
        """
        enhanced_parts = []
        
        # Add role if specified
        if self.config.add_role and role:
            enhanced_parts.append(self._add_role(role))
        
        # Add context
        if context:
            enhanced_parts.append(self._add_context(context))
        
        # Enhance clarity and structure
        clear_prompt = self._enhance_clarity(original_prompt)
        
        # Add task decomposition for complex prompts
        if self._is_complex_task(clear_prompt):
            clear_prompt = self._decompose_task(clear_prompt)
        
        # Structure with XML tags if enabled
        if self.config.add_xml_tags:
            clear_prompt = self._add_xml_structure(clear_prompt, task_type)
        
        enhanced_parts.append(clear_prompt)
        
        # Add examples if provided
        if self.config.add_examples and examples:
            enhanced_parts.append(self._format_examples(examples))
        
        # Add chain of thought if enabled
        if self.config.add_chain_of_thought:
            enhanced_parts.append(self._add_chain_of_thought(task_type))
        
        # Add output format specifications
        if self.config.add_output_format and desired_format:
            enhanced_parts.append(self._specify_output_format(desired_format))
        
        # Add constraints
        if self.config.add_constraints:
            enhanced_parts.append(self._add_constraints(task_type))
        
        # Combine all parts
        enhanced_prompt = "\n\n".join(filter(None, enhanced_parts))
        
        # Apply length constraint if specified
        if self.config.max_length:
            enhanced_prompt = self._trim_to_length(enhanced_prompt, self.config.max_length)
        
        return enhanced_prompt
    
    def _enhance_clarity(self, prompt: str) -> str:
        """Enhance prompt clarity by improving structure and specificity"""
        # Remove excessive whitespace
        prompt = re.sub(r'\s+', ' ', prompt.strip())
        
        # Ensure proper punctuation
        if prompt and not prompt[-1] in '.?!':
            prompt += '.'
        
        # Make instructions more explicit
        clarity_patterns = [
            (r'\b(do|make|create)\b', r'Please \1'),
            (r'^(\w)', lambda m: m.group(1).upper()),  # Capitalize first letter
            (r'\bit\b', 'the requested output'),  # Replace vague "it"
        ]
        
        for pattern, replacement in clarity_patterns:
            prompt = re.sub(pattern, replacement, prompt)
        
        return prompt
    
    def _add_xml_structure(self, prompt: str, task_type: Optional[str]) -> str:
        """Add XML tags for better structure"""
        structured = f"<task>\n{prompt}\n</task>"
        
        if task_type:
            structured = f"<task_type>{task_type}</task_type>\n{structured}"
        
        return structured
    
    def _add_role(self, role: str) -> str:
        """Add role or persona specification"""
        return f"<role>\nYou are {role}. Respond accordingly with expertise and perspective appropriate to this role.\n</role>"
    
    def _add_context(self, context: str) -> str:
        """Add context section"""
        return f"<context>\n{context}\n</context>"
    
    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format examples in a clear structure"""
        if not examples:
            return ""
        
        formatted = "<examples>\nHere are some examples:\n\n"
        
        for i, example in enumerate(examples, 1):
            formatted += f"Example {i}:\n"
            if "input" in example:
                formatted += f"<input>\n{example['input']}\n</input>\n"
            if "output" in example:
                formatted += f"<output>\n{example['output']}\n</output>\n"
            formatted += "\n"
        
        formatted += "</examples>"
        return formatted
    
    def _add_chain_of_thought(self, task_type: Optional[str]) -> str:
        """Add chain of thought reasoning instruction"""
        cot_instruction = "<thinking>\nBefore providing your answer, please think through this step-by-step:\n"
        
        if task_type == "analysis":
            cot_instruction += "1. Identify key components\n2. Analyze relationships\n3. Draw conclusions\n"
        elif task_type == "problem_solving":
            cot_instruction += "1. Understand the problem\n2. Identify constraints\n3. Develop solution approach\n4. Verify solution\n"
        elif task_type == "code_generation":
            cot_instruction += "1. Understand requirements\n2. Plan the implementation\n3. Consider edge cases\n4. Write clean code\n"
        else:
            cot_instruction += "1. Break down the request\n2. Consider different approaches\n3. Select the best approach\n"
        
        cot_instruction += "</thinking>"
        return cot_instruction
    
    def _specify_output_format(self, format_spec: str) -> str:
        """Specify desired output format"""
        return f"<output_format>\n{format_spec}\n</output_format>"
    
    def _add_constraints(self, task_type: Optional[str]) -> str:
        """Add relevant constraints based on task type"""
        constraints = ["<constraints>"]
        
        # General constraints
        constraints.append("- Be accurate and precise")
        constraints.append("- Provide complete information")
        
        # Task-specific constraints
        if task_type == "code_generation":
            constraints.extend([
                "- Include error handling",
                "- Add comments for complex logic",
                "- Follow best practices and conventions"
            ])
        elif task_type == "analysis":
            constraints.extend([
                "- Support claims with evidence",
                "- Consider multiple perspectives",
                "- Be objective and unbiased"
            ])
        elif task_type == "creative_writing":
            constraints.extend([
                "- Be original and engaging",
                "- Maintain consistent tone and style",
                "- Show don't tell"
            ])
        
        constraints.append("</constraints>")
        return "\n".join(constraints)
    
    def _is_complex_task(self, prompt: str) -> bool:
        """Determine if a task is complex enough to need decomposition"""
        indicators = [
            len(prompt.split('.')) > 3,
            len(prompt.split(',')) > 4,
            any(word in prompt.lower() for word in ['and then', 'after that', 'finally', 'multiple']),
            len(prompt.split()) > 50
        ]
        return sum(indicators) >= 2
    
    def _decompose_task(self, prompt: str) -> str:
        """Break down complex tasks into steps"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', prompt) if s.strip()]
        
        if len(sentences) > 1:
            decomposed = "<task_steps>\n"
            for i, sentence in enumerate(sentences, 1):
                decomposed += f"Step {i}: {sentence}\n"
            decomposed += "</task_steps>"
            return decomposed
        
        return prompt
    
    def _trim_to_length(self, prompt: str, max_length: int) -> str:
        """Trim prompt to maximum length while preserving structure"""
        if len(prompt) <= max_length:
            return prompt
        
        sentences = prompt.split('. ')
        trimmed = ""
        
        for sentence in sentences:
            if len(trimmed) + len(sentence) + 2 <= max_length:
                trimmed += sentence + ". "
            else:
                break
        
        return trimmed.strip()

def test_prompt_with_api(prompt: str) -> dict:
    """Test the enhanced prompt with the Claude API"""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": OPENAI_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
def main():
    st.title("🚀 Prompt Enhancer Pro")
    st.markdown("Enhance your prompts using best practices from Anthropic and OpenAI")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Enhancement Configuration")
        
        task_type = st.selectbox(
            "Task Type",
            ["general", "code_generation", "analysis", "creative_writing", "problem_solving"],
            help="Select the type of task for optimized enhancement"
        )
        
        st.subheader("Enhancement Options")
        add_xml_tags = st.checkbox("Add XML Structure", value=True, help="Structure prompt with XML tags")
        add_chain_of_thought = st.checkbox("Add Chain of Thought", value=False, help="Add step-by-step thinking")
        add_role = st.checkbox("Add Role/Persona", value=False, help="Specify a role for the AI")
        add_constraints = st.checkbox("Add Constraints", value=True, help="Add task-specific constraints")
        add_output_format = st.checkbox("Specify Output Format", value=True, help="Define expected output format")
        add_examples = st.checkbox("Include Examples", value=False, help="Add input-output examples")
        
        max_length = st.number_input(
            "Max Length (optional)",
            min_value=0,
            value=0,
            help="Set to 0 for no limit"
        )
        
        test_with_api = st.checkbox("Test Enhanced Prompt", value=False, help="Test the enhanced prompt with Claude API")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Original Prompt")
        original_prompt = st.text_area(
            "Enter your prompt",
            height=200,
            placeholder="Example: Create a Python function to analyze customer data and generate insights"
        )
        
        # Additional inputs based on configuration
        role_input = None
        if add_role:
            role_input = st.text_input(
                "Role Description",
                placeholder="e.g., expert data scientist with 10 years of experience"
            )
        
        context_input = st.text_area(
            "Additional Context (optional)",
            height=100,
            placeholder="Any additional context or background information"
        )
        
        output_format_input = None
        if add_output_format:
            output_format_input = st.text_area(
                "Desired Output Format",
                height=100,
                placeholder="e.g., Provide a structured response with: 1. Summary 2. Detailed Analysis 3. Recommendations"
            )
        
        # Examples input
        examples = []
        if add_examples:
            st.markdown("#### Examples")
            num_examples = st.number_input("Number of examples", min_value=0, max_value=5, value=1)
            
            for i in range(num_examples):
                with st.expander(f"Example {i+1}"):
                    example_input = st.text_area(f"Input {i+1}", key=f"ex_input_{i}")
                    example_output = st.text_area(f"Output {i+1}", key=f"ex_output_{i}")
                    if example_input and example_output:
                        examples.append({"input": example_input, "output": example_output})
    
    with col2:
        st.subheader("✨ Enhanced Prompt")
        
        if st.button("🔧 Enhance Prompt", type="primary", use_container_width=True):
            if original_prompt:
                # Create configuration
                config = EnhancementConfig(
                    add_xml_tags=add_xml_tags,
                    add_examples=add_examples and len(examples) > 0,
                    add_constraints=add_constraints,
                    add_output_format=add_output_format and output_format_input,
                    add_chain_of_thought=add_chain_of_thought,
                    add_role=add_role and role_input,
                    max_length=max_length if max_length > 0 else None
                )
                
                # Enhance the prompt
                enhancer = PromptEnhancer(config)
                enhanced_prompt = enhancer.enhance_prompt(
                    original_prompt,
                    task_type=task_type,
                    examples=examples if examples else None,
                    context=context_input if context_input else None,
                    desired_format=output_format_input if output_format_input else None,
                    role=role_input if role_input else None
                )
                
                # Store in session state
                st.session_state['enhanced_prompt'] = enhanced_prompt
                st.session_state['enhancement_timestamp'] = datetime.now()
            else:
                st.error("Please enter a prompt to enhance")
        
        # Display enhanced prompt
        if 'enhanced_prompt' in st.session_state:
            st.text_area(
                "Enhanced Prompt",
                value=st.session_state['enhanced_prompt'],
                height=400,
                key="enhanced_display"
            )
            
            # Copy button
            st.button(
                "📋 Copy to Clipboard",
                on_click=lambda: st.write("Copied!"),
                help="Copy the enhanced prompt to clipboard"
            )
            
            # Download button
            st.download_button(
                label="📥 Download Enhanced Prompt",
                data=st.session_state['enhanced_prompt'],
                file_name=f"enhanced_prompt_{st.session_state['enhancement_timestamp'].strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    # Test with API section
    if test_with_api and 'enhanced_prompt' in st.session_state:
        st.markdown("---")
        st.subheader("🧪 API Test Results")
        
        if st.button("🚀 Test Enhanced Prompt with Claude", use_container_width=True):
            with st.spinner("Testing prompt with Claude API..."):
                result = test_prompt_with_api(st.session_state['enhanced_prompt'])
                
                if "error" in result:
                    st.error(f"API Error: {result['error']}")
                else:
                    st.success("API call successful!")
                    
                    # Display response
                    if "choices" in result and len(result["choices"]) > 0:
                        response_content = result["choices"][0]["message"]["content"]
                        st.markdown("#### Claude's Response:")
                        st.markdown(response_content)
                        
                        # Show token usage if available
                        if "usage" in result:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Prompt Tokens", result["usage"].get("prompt_tokens", "N/A"))
                            with col2:
                                st.metric("Completion Tokens", result["usage"].get("completion_tokens", "N/A"))
                            with col3:
                                st.metric("Total Tokens", result["usage"].get("total_tokens", "N/A"))
    
    # Examples section
    with st.expander("📚 Example Enhancements"):
        st.markdown("""
        ### Example 1: Code Generation
        **Original:** "make a function to sort a list"
        
        **Enhanced:** Includes XML structure, clear specifications, error handling requirements, and output format.
        
        ### Example 2: Analysis
        **Original:** "analyze this data"
        
        **Enhanced:** Adds role specification, chain of thought reasoning, structured output format, and analytical constraints.
        
        ### Example 3: Creative Writing
        **Original:** "write a story"
        
        **Enhanced:** Specifies role, adds creative constraints, defines style and tone, and includes structural guidelines.
        """)
    
    # Tips section
    with st.expander("💡 Pro Tips"):
        st.markdown("""
        1. **Be Specific:** The more details you provide, the better the enhancement
        2. **Use Examples:** When applicable, provide examples for few-shot learning
        3. **Define Output:** Clearly specify the format you want for the response
        4. **Add Context:** Background information helps create more targeted prompts
        5. **Test Iteratively:** Use the API test feature to refine your prompts
        """)

if __name__ == "__main__":
    main()