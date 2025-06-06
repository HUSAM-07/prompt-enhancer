import streamlit as st
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import requests
import json
from datetime import datetime
import PyPDF2
import io
import csv
import pandas as pd
import uuid
from collections import defaultdict

# Configure page
st.set_page_config(
    page_title="Prompt Enhancement Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []
if 'examples' not in st.session_state:
    st.session_state.examples = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = ""
if 'enhanced_prompt' not in st.session_state:
    st.session_state.enhanced_prompt = ""
if 'improvement_feedback' not in st.session_state:
    st.session_state.improvement_feedback = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "https://api.openai.com/v1"
if 'model_name' not in st.session_state:
    st.session_state.model_name = "gpt-4"
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False

@dataclass
class Example:
    """Structured example for few-shot learning"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    input: str = ""
    output: str = ""
    chain_of_thought: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class EvaluationResult:
    """Result of prompt evaluation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt_version: str = ""
    test_input: str = ""
    model_output: str = ""
    ideal_output: str = ""
    score: int = 0  # 1-5 scale
    feedback: str = ""
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PromptVersion:
    """Version of a prompt with metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    techniques_used: List[str] = field(default_factory=list)
    improvement_notes: str = ""
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class PromptImprovementEngine:
    """Advanced prompt improvement engine using Claude"""
    
    def __init__(self):
        self.improvement_prompt_template = """You are an expert prompt engineer specializing in optimizing prompts for LLMs.

Your task is to improve the following prompt using these advanced techniques:

1. **Chain-of-thought reasoning**: Add a dedicated <thinking> section for systematic problem-solving
2. **Example standardization**: Convert all examples to consistent XML format with clear input/output pairs
3. **Example enrichment**: Add chain-of-thought reasoning to examples that aligns with the prompt structure
4. **Clarity rewriting**: Improve structure, fix grammar/spelling, and enhance clarity
5. **Prefill addition**: Add assistant message prefill to guide output format
6. **XML structuring**: Use clear XML tags for different sections

<original_prompt>
{original_prompt}
</original_prompt>

<current_examples>
{examples}
</current_examples>

<improvement_feedback>
{feedback}
</improvement_feedback>

Please provide:
1. The improved prompt with all enhancements
2. A list of specific improvements made
3. Suggestions for additional examples that would help

Format your response as:
<improved_prompt>
[Your enhanced prompt here]
</improved_prompt>

<improvements_made>
- [List each improvement]
</improvements_made>

<example_suggestions>
- [Suggest helpful examples]
</example_suggestions>"""

    def improve_prompt(self, 
                      original_prompt: str, 
                      examples: List[Example] = None,
                      feedback: List[str] = None) -> Tuple[str, List[str], List[str]]:
        """Use AI to automatically improve a prompt"""
        
        # Format examples for the improvement prompt
        examples_text = self._format_examples_for_improvement(examples or [])
        feedback_text = "\n".join(feedback or ["No specific feedback provided"])
        
        # Create the improvement request
        improvement_request = self.improvement_prompt_template.format(
            original_prompt=original_prompt,
            examples=examples_text,
            feedback=feedback_text
        )
        
        # Call AI API
        response = self._call_claude_api(improvement_request)
        
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
            
            # Parse the response
            improved_prompt = self._extract_section(content, "improved_prompt")
            improvements = self._extract_list_section(content, "improvements_made")
            suggestions = self._extract_list_section(content, "example_suggestions")
            
            return improved_prompt, improvements, suggestions
        
        return original_prompt, ["Error: Could not improve prompt"], []
    
    def _format_examples_for_improvement(self, examples: List[Example]) -> str:
        """Format examples for the improvement prompt"""
        if not examples:
            return "No examples provided"
        
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"Input: {ex.input}")
            formatted.append(f"Output: {ex.output}")
            if ex.chain_of_thought:
                formatted.append(f"Chain of thought: {ex.chain_of_thought}")
            formatted.append("")
        
        return "\n".join(formatted)
    
    def _call_claude_api(self, prompt: str) -> dict:
        """Call LLM API with the prompt"""
        if not st.session_state.api_configured:
            st.error("Please configure your API settings in the sidebar first!")
            return None
            
        headers = {
            "Authorization": f"Bearer {st.session_state.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": st.session_state.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000,
            "temperature": 0.3  # Lower temperature for more consistent improvements
        }
        
        try:
            response = requests.post(
                f"{st.session_state.api_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            return None
    
    def _extract_section(self, content: str, tag: str) -> str:
        """Extract content between XML tags"""
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""
    
    def _extract_list_section(self, content: str, tag: str) -> List[str]:
        """Extract list items from a section"""
        section = self._extract_section(content, tag)
        if not section:
            return []
        
        # Extract items that start with - or *
        items = re.findall(r'[-*]\s*(.+)', section)
        return [item.strip() for item in items]

class ExampleManager:
    """Manage multi-shot examples with enrichment"""
    
    def __init__(self):
        self.enrichment_prompt = """Given this example, add chain-of-thought reasoning that shows how to arrive at the output from the input.

<example>
Input: {input}
Output: {output}
</example>

Provide a clear, step-by-step chain of thought that explains the reasoning process.
Format as: <chain_of_thought>[Your reasoning here]</chain_of_thought>"""
    
    def enrich_example(self, example: Example) -> Example:
        """Enrich an example with chain-of-thought reasoning"""
        if example.chain_of_thought:
            return example
        
        prompt = self.enrichment_prompt.format(
            input=example.input,
            output=example.output
        )
        
        # Call AI to generate chain of thought
        if not st.session_state.api_configured:
            return example
            
        headers = {
            "Authorization": f"Bearer {st.session_state.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": st.session_state.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        try:
            response = requests.post(
                f"{st.session_state.api_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result:
                content = result["choices"][0]["message"]["content"]
                # Extract chain of thought
                pattern = r"<chain_of_thought>(.*?)</chain_of_thought>"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    example.chain_of_thought = match.group(1).strip()
        except Exception as e:
            st.error(f"Error enriching example: {str(e)}")
        
        return example
    
    def format_examples_xml(self, examples: List[Example]) -> str:
        """Format examples in standardized XML format"""
        if not examples:
            return ""
        
        xml_parts = ["<examples>"]
        
        for i, ex in enumerate(examples, 1):
            xml_parts.append(f"  <example id=\"{i}\">")
            xml_parts.append(f"    <input>{ex.input}</input>")
            xml_parts.append(f"    <output>{ex.output}</output>")
            if ex.chain_of_thought:
                xml_parts.append(f"    <reasoning>{ex.chain_of_thought}</reasoning>")
            xml_parts.append("  </example>")
        
        xml_parts.append("</examples>")
        return "\n".join(xml_parts)

class PromptEvaluator:
    """Evaluate prompts with test cases and ideal outputs"""
    
    def __init__(self):
        self.grading_criteria = {
            5: "Perfect match - Output matches ideal output exactly or exceeds expectations",
            4: "Good match - Output is very close to ideal with minor differences",
            3: "Acceptable - Output captures main points but missing some details",
            2: "Poor match - Output misses significant aspects of ideal output",
            1: "Failed - Output is completely wrong or irrelevant"
        }
    
    def evaluate_prompt(self, 
                       prompt: str, 
                       test_input: str,
                       ideal_output: str = None) -> EvaluationResult:
        """Evaluate a prompt with a test case"""
        
        # Prepare the test prompt
        test_prompt = prompt.replace("{{input}}", test_input)
        
        # Get model output
        if not st.session_state.api_configured:
            st.error("Please configure your API settings in the sidebar first!")
            return None
            
        headers = {
            "Authorization": f"Bearer {st.session_state.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": st.session_state.model_name,
            "messages": [{"role": "user", "content": test_prompt}],
            "max_tokens": 2000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                f"{st.session_state.api_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result:
                model_output = result["choices"][0]["message"]["content"]
                
                # Create evaluation result
                eval_result = EvaluationResult(
                    prompt_version=prompt[:100] + "...",  # Store prompt preview
                    test_input=test_input,
                    model_output=model_output,
                    ideal_output=ideal_output or "",
                    score=0,
                    feedback=""
                )
                
                # Auto-grade if ideal output provided
                if ideal_output:
                    eval_result.score = self._auto_grade(model_output, ideal_output)
                
                return eval_result
                
        except Exception as e:
            st.error(f"Evaluation error: {str(e)}")
            return None
    
    def _auto_grade(self, model_output: str, ideal_output: str) -> int:
        """Automatically grade output against ideal (simplified)"""
        # This is a simplified grading - in production, use more sophisticated comparison
        model_lower = model_output.lower().strip()
        ideal_lower = ideal_output.lower().strip()
        
        # Exact match
        if model_lower == ideal_lower:
            return 5
        
        # Check key phrases
        ideal_words = set(ideal_lower.split())
        model_words = set(model_lower.split())
        overlap = len(ideal_words & model_words) / len(ideal_words)
        
        if overlap > 0.8:
            return 4
        elif overlap > 0.6:
            return 3
        elif overlap > 0.4:
            return 2
        else:
            return 1

def main():
    st.title("🚀 Prompt Enhancement Studio")
    st.markdown("Advanced prompt engineering with automatic improvement, example management, and evaluation for any LLM")
    
    # Initialize engines
    improvement_engine = PromptImprovementEngine()
    example_manager = ExampleManager()
    evaluator = PromptEvaluator()
    
    # Sidebar
    with st.sidebar:
        # API Configuration Section
        st.header("🔑 API Configuration")
        
        with st.expander("API Settings", expanded=not st.session_state.api_configured):
            # API Provider selection
            api_provider = st.selectbox(
                "API Provider",
                ["OpenAI", "Anthropic (Claude)", "Custom/LiteLLM"],
                key="api_provider_select"
            )
            
            # Set default values based on provider
            if api_provider == "OpenAI":
                default_base_url = "https://api.openai.com/v1"
                default_models = [
                    "gpt-4.1",
                    "gpt-4o",
                    "chatgpt-4o-latest",
                    "o4-mini",
                    "gpt-4.1-mini",
                    "gpt-4.1-nano",
                    "o3-mini",
                    "gpt-4o-mini",
                    "o3",
                    "o1",
                    "o1-mini",
                    "o1-pro"
                ]
            elif api_provider == "Anthropic (Claude)":
                default_base_url = "https://api.anthropic.com/v1"
                default_models = [
                    "claude-opus-4-0",
                    "claude-sonnet-4-0",
                    "claude-3-7-sonnet-latest",
                    "claude-3-5-sonnet-latest",
                    "claude-3-5-haiku-latest",
                    "claude-3-opus-latest"
                ]
            else:
                default_base_url = st.session_state.api_base_url
                default_models = ["claude-4-sonnet", "gpt-4", "custom-model"]
            
            # API Key input
            api_key = st.text_input(
                "API Key",
                value=st.session_state.api_key,
                type="password",
                placeholder="sk-...",
                help="Your API key will be stored only for this session"
            )
            
            # API Base URL
            api_base_url = st.text_input(
                "API Base URL",
                value=default_base_url,
                placeholder="https://api.openai.com/v1",
                help="The base URL for the API endpoint"
            )
            
            # Model selection
            if api_provider == "Custom/LiteLLM":
                model_name = st.text_input(
                    "Model Name",
                    value=st.session_state.model_name,
                    placeholder="Enter your model name",
                    help="The exact model name to use"
                )
            else:
                # Create display names for models
                if api_provider == "OpenAI":
                    model_display_names = {
                        "gpt-4.1": "GPT-4.1 (Flagship)",
                        "gpt-4o": "GPT-4o (Fast & Flexible)",
                        "chatgpt-4o-latest": "ChatGPT-4o (Latest)",
                        "o4-mini": "o4-mini (Affordable Reasoning)",
                        "gpt-4.1-mini": "GPT-4.1 mini (Balanced)",
                        "gpt-4.1-nano": "GPT-4.1 nano (Fastest)",
                        "o3-mini": "o3-mini (Small Alternative)",
                        "gpt-4o-mini": "GPT-4o mini (Fast & Affordable)",
                        "o3": "o3 (Most Powerful Reasoning)",
                        "o1": "o1 (Full Reasoning)",
                        "o1-mini": "o1-mini (Small Reasoning)",
                        "o1-pro": "o1-pro (Enhanced Reasoning)"
                    }
                    display_options = [model_display_names.get(m, m) for m in default_models]
                    selected_display = st.selectbox(
                        "Model",
                        display_options,
                        index=0 if not st.session_state.model_name else (
                            default_models.index(st.session_state.model_name) 
                            if st.session_state.model_name in default_models else 0
                        ),
                        help="Select from OpenAI's latest models"
                    )
                    # Get the actual model ID from the display name
                    model_name = default_models[display_options.index(selected_display)]
                elif api_provider == "Anthropic (Claude)":
                    model_display_names = {
                        "claude-opus-4-0": "Claude Opus 4",
                        "claude-sonnet-4-0": "Claude Sonnet 4",
                        "claude-3-7-sonnet-latest": "Claude Sonnet 3.7",
                        "claude-3-5-sonnet-latest": "Claude Sonnet 3.5",
                        "claude-3-5-haiku-latest": "Claude Haiku 3.5",
                        "claude-3-opus-latest": "Claude Opus 3"
                    }
                    display_options = [model_display_names.get(m, m) for m in default_models]
                    selected_display = st.selectbox(
                        "Model",
                        display_options,
                        index=0 if not st.session_state.model_name else (
                            default_models.index(st.session_state.model_name) 
                            if st.session_state.model_name in default_models else 0
                        )
                    )
                    # Get the actual model ID from the display name
                    model_name = default_models[display_options.index(selected_display)]
                else:
                    model_name = st.selectbox(
                        "Model",
                        default_models,
                        index=0 if not st.session_state.model_name else (
                            default_models.index(st.session_state.model_name) 
                            if st.session_state.model_name in default_models else 0
                        )
                    )
            
            # Save configuration button
            if st.button("💾 Save Configuration", type="primary", use_container_width=True):
                if api_key and api_base_url and model_name:
                    st.session_state.api_key = api_key
                    st.session_state.api_base_url = api_base_url
                    st.session_state.model_name = model_name
                    st.session_state.api_configured = True
                    st.success("✅ API configuration saved!")
                    st.rerun()
                else:
                    st.error("Please fill in all required fields")
            
            # Test connection button
            if st.session_state.api_configured:
                if st.button("🧪 Test Connection", use_container_width=True):
                    with st.spinner("Testing API connection..."):
                        headers = {
                            "Authorization": f"Bearer {st.session_state.api_key}",
                            "Content-Type": "application/json"
                        }
                        
                        test_data = {
                            "model": st.session_state.model_name,
                            "messages": [{"role": "user", "content": "Hello, this is a test."}],
                            "max_tokens": 10
                        }
                        
                        try:
                            response = requests.post(
                                f"{st.session_state.api_base_url}/chat/completions",
                                headers=headers,
                                json=test_data,
                                timeout=10
                            )
                            response.raise_for_status()
                            st.success("✅ Connection successful!")
                        except Exception as e:
                            st.error(f"❌ Connection failed: {str(e)}")
        
        # Show current configuration status
        if st.session_state.api_configured:
            # Show friendly name for models
            model_display_names = {
                # OpenAI models
                "gpt-4.1": "GPT-4.1 (Flagship)",
                "gpt-4o": "GPT-4o (Fast & Flexible)",
                "chatgpt-4o-latest": "ChatGPT-4o (Latest)",
                "o4-mini": "o4-mini (Affordable Reasoning)",
                "gpt-4.1-mini": "GPT-4.1 mini (Balanced)",
                "gpt-4.1-nano": "GPT-4.1 nano (Fastest)",
                "o3-mini": "o3-mini (Small Alternative)",
                "gpt-4o-mini": "GPT-4o mini (Fast & Affordable)",
                "o3": "o3 (Most Powerful Reasoning)",
                "o1": "o1 (Full Reasoning)",
                "o1-mini": "o1-mini (Small Reasoning)",
                "o1-pro": "o1-pro (Enhanced Reasoning)",
                # Anthropic models
                "claude-opus-4-0": "Claude Opus 4",
                "claude-sonnet-4-0": "Claude Sonnet 4",
                "claude-3-7-sonnet-latest": "Claude Sonnet 3.7",
                "claude-3-5-sonnet-latest": "Claude Sonnet 3.5",
                "claude-3-5-haiku-latest": "Claude Haiku 3.5",
                "claude-3-opus-latest": "Claude Opus 3"
            }
            display_model = model_display_names.get(st.session_state.model_name, st.session_state.model_name)
            st.success(f"✅ Connected to {display_model}")
        else:
            st.warning("⚠️ Please configure API settings above")
        
        st.divider()
        
        st.header("📚 Quick Guide")
        st.markdown("""
        1. **Configure API** settings above
        2. **Enter your prompt** in the main area
        3. **Add examples** to improve accuracy
        4. **Auto-improve** using AI expertise
        5. **Evaluate** with test cases
        6. **Iterate** based on feedback
        
        ### Features:
        - ✨ Automatic prompt improvement
        - 🔄 Chain-of-thought reasoning
        - 📝 Example standardization
        - 🎯 Evaluation with ideal outputs
        - 📊 Performance tracking
        """)
        
        st.divider()
        
        # Prompt history
        if st.session_state.prompt_history:
            st.header("📜 Prompt History")
            for i, version in enumerate(reversed(st.session_state.prompt_history[-5:])):
                with st.expander(f"Version {len(st.session_state.prompt_history) - i}"):
                    st.text(version.content[:200] + "...")
                    st.caption(f"Created: {version.created_at.strftime('%Y-%m-%d %H:%M')}")
    
    # Check if API is configured
    if not st.session_state.api_configured:
        st.warning("⚠️ Please configure your API settings in the sidebar to get started.")
        st.info("👈 Click on 'API Settings' in the sidebar and enter your API key and model details.")
        st.stop()
    
    # Main content area with tabs
    tab1, tab2, tab3, tab4 = st.tabs(["✏️ Prompt Editor", "📚 Examples", "🧪 Evaluation", "📊 Analytics"])
    
    with tab1:
        st.header("Prompt Enhancement")
        
        # Original prompt input
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Prompt")
            original_prompt = st.text_area(
                "Enter your prompt",
                value=st.session_state.current_prompt,
                height=300,
                placeholder="Enter your prompt here. Use {{input}} for variable substitution.",
                key="original_prompt_input"
            )
            
            # Improvement feedback
            st.subheader("Improvement Feedback (Optional)")
            feedback = st.text_area(
                "What aspects need improvement?",
                height=100,
                placeholder="e.g., 'Output is too verbose', 'Needs better structure', 'Missing error handling'",
                key="improvement_feedback_input"
            )
            
            # Action buttons
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🚀 Auto-Improve Prompt", type="primary", use_container_width=True):
                    if original_prompt:
                        with st.spinner("AI is improving your prompt..."):
                            # Add feedback to history
                            if feedback:
                                st.session_state.improvement_feedback.append(feedback)
                            
                            # Improve the prompt
                            improved, improvements, suggestions = improvement_engine.improve_prompt(
                                original_prompt,
                                st.session_state.examples,
                                st.session_state.improvement_feedback
                            )
                            
                            # Update session state
                            st.session_state.enhanced_prompt = improved
                            st.session_state.current_prompt = original_prompt
                            
                            # Add to history
                            version = PromptVersion(
                                content=improved,
                                techniques_used=improvements,
                                improvement_notes=feedback
                            )
                            st.session_state.prompt_history.append(version)
                            
                            # Show improvements
                            with st.expander("✨ Improvements Made", expanded=True):
                                for imp in improvements:
                                    st.write(f"• {imp}")
                            
                            # Show suggestions
                            if suggestions:
                                with st.expander("💡 Example Suggestions"):
                                    for sug in suggestions:
                                        st.write(f"• {sug}")
                    else:
                        st.error("Please enter a prompt to improve")
            
            with col_btn2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.current_prompt = ""
                    st.session_state.enhanced_prompt = ""
                    st.session_state.improvement_feedback = []
                    st.rerun()
        
        with col2:
            st.subheader("Enhanced Prompt")
            
            if st.session_state.enhanced_prompt:
                # Display enhanced prompt
                st.text_area(
                    "Improved version",
                    value=st.session_state.enhanced_prompt,
                    height=400,
                    key="enhanced_prompt_display"
                )
                
                # Copy and download buttons
                col_copy, col_download = st.columns(2)
                
                with col_copy:
                    if st.button("📋 Copy Enhanced", use_container_width=True):
                        st.success("Copied to clipboard!")
                
                with col_download:
                    st.download_button(
                        "📥 Download",
                        data=st.session_state.enhanced_prompt,
                        file_name=f"enhanced_prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            else:
                st.info("👈 Click 'Auto-Improve Prompt' to see the enhanced version")
    
    with tab2:
        st.header("Example Management")
        
        # Add new example
        with st.expander("➕ Add New Example", expanded=True):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                example_input = st.text_area("Input", height=100, key="new_example_input")
            
            with col2:
                example_output = st.text_area("Expected Output", height=100, key="new_example_output")
            
            col3, col4 = st.columns([3, 1])
            
            with col3:
                example_tags = st.text_input("Tags (comma-separated)", key="new_example_tags")
            
            with col4:
                if st.button("Add Example", type="primary", use_container_width=True):
                    if example_input and example_output:
                        new_example = Example(
                            input=example_input,
                            output=example_output,
                            tags=[tag.strip() for tag in example_tags.split(",")] if example_tags else []
                        )
                        
                        # Auto-enrich with chain of thought
                        with st.spinner("Enriching example with chain-of-thought..."):
                            enriched = example_manager.enrich_example(new_example)
                        
                        st.session_state.examples.append(enriched)
                        st.success("Example added and enriched!")
                        st.rerun()
                    else:
                        st.error("Please provide both input and output")
        
        # Display existing examples
        if st.session_state.examples:
            st.subheader(f"Current Examples ({len(st.session_state.examples)})")
            
            # Show XML format
            with st.expander("📄 View XML Format"):
                xml_examples = example_manager.format_examples_xml(st.session_state.examples)
                st.code(xml_examples, language="xml")
            
            # Individual examples
            for i, example in enumerate(st.session_state.examples):
                with st.expander(f"Example {i+1}: {example.input[:50]}..."):
                    col1, col2 = st.columns([5, 1])
                    
                    with col1:
                        st.markdown("**Input:**")
                        st.text(example.input)
                        
                        st.markdown("**Output:**")
                        st.text(example.output)
                        
                        if example.chain_of_thought:
                            st.markdown("**Chain of Thought:**")
                            st.info(example.chain_of_thought)
                        
                        if example.tags:
                            st.markdown(f"**Tags:** {', '.join(example.tags)}")
                    
                    with col2:
                        if st.button("🗑️ Delete", key=f"delete_example_{i}"):
                            st.session_state.examples.pop(i)
                            st.rerun()
        else:
            st.info("No examples yet. Add some to improve prompt accuracy!")
    
    with tab3:
        st.header("Prompt Evaluation")
        
        # Test configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            test_input = st.text_area(
                "Test Input",
                height=100,
                placeholder="Enter test input to evaluate the prompt",
                key="test_input"
            )
        
        with col2:
            include_ideal = st.checkbox("Include Ideal Output", value=True)
        
        if include_ideal:
            ideal_output = st.text_area(
                "Ideal Output (for automatic grading)",
                height=100,
                placeholder="Enter the expected ideal output",
                key="ideal_output"
            )
        else:
            ideal_output = None
        
        # Evaluation button
        if st.button("🧪 Evaluate Prompt", type="primary", use_container_width=True):
            if test_input and st.session_state.enhanced_prompt:
                with st.spinner("Evaluating prompt..."):
                    result = evaluator.evaluate_prompt(
                        st.session_state.enhanced_prompt,
                        test_input,
                        ideal_output
                    )
                    
                    if result:
                        st.session_state.evaluation_results.append(result)
                        
                        # Display result
                        st.success("Evaluation complete!")
                        
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown("**Model Output:**")
                            st.text_area("", value=result.model_output, height=200, disabled=True)
                        
                        with col2:
                            if result.score > 0:
                                st.metric("Auto Score", f"{result.score}/5")
                                st.caption(evaluator.grading_criteria[result.score])
                        
                        # Manual grading
                        st.markdown("**Manual Grading:**")
                        col1, col2, col3 = st.columns([1, 2, 2])
                        
                        with col1:
                            manual_score = st.select_slider(
                                "Score",
                                options=[1, 2, 3, 4, 5],
                                value=result.score if result.score > 0 else 3,
                                key="manual_score"
                            )
                        
                        with col2:
                            feedback = st.text_input("Feedback", key="eval_feedback")
                        
                        with col3:
                            if st.button("Save Evaluation"):
                                result.score = manual_score
                                result.feedback = feedback
                                st.success("Evaluation saved!")
            else:
                st.error("Please provide test input and ensure you have an enhanced prompt")
        
        # Evaluation history
        if st.session_state.evaluation_results:
            st.divider()
            st.subheader("Evaluation History")
            
            # Summary metrics
            scores = [r.score for r in st.session_state.evaluation_results if r.score > 0]
            if scores:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Score", f"{sum(scores)/len(scores):.1f}/5")
                with col2:
                    st.metric("Total Evaluations", len(st.session_state.evaluation_results))
                with col3:
                    st.metric("Success Rate", f"{len([s for s in scores if s >= 3])/len(scores)*100:.0f}%")
            
            # Individual results
            for i, result in enumerate(reversed(st.session_state.evaluation_results[-5:])):
                with st.expander(f"Test {len(st.session_state.evaluation_results) - i}"):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**Input:** {result.test_input}")
                        st.markdown(f"**Output:** {result.model_output[:200]}...")
                        if result.ideal_output:
                            st.markdown(f"**Ideal:** {result.ideal_output[:200]}...")
                        if result.feedback:
                            st.markdown(f"**Feedback:** {result.feedback}")
                    
                    with col2:
                        if result.score > 0:
                            st.metric("Score", f"{result.score}/5")
    
    with tab4:
        st.header("Performance Analytics")
        
        if st.session_state.prompt_history and st.session_state.evaluation_results:
            # Performance over time
            st.subheader("📈 Performance Trends")
            
            # Create performance data
            eval_data = []
            for result in st.session_state.evaluation_results:
                if result.score > 0:
                    eval_data.append({
                        'Timestamp': result.created_at,
                        'Score': result.score,
                        'Test': result.test_input[:30] + "..."
                    })
            
            if eval_data:
                df = pd.DataFrame(eval_data)
                st.line_chart(df.set_index('Timestamp')['Score'])
            
            # Technique effectiveness
            st.subheader("🎯 Technique Effectiveness")
            
            technique_scores = defaultdict(list)
            for version in st.session_state.prompt_history:
                # Find evaluations for this version
                version_evals = [e for e in st.session_state.evaluation_results 
                               if e.prompt_version == version.content[:100] + "..."]
                
                if version_evals:
                    avg_score = sum(e.score for e in version_evals if e.score > 0) / len(version_evals)
                    for technique in version.techniques_used:
                        technique_scores[technique].append(avg_score)
            
            if technique_scores:
                technique_df = pd.DataFrame([
                    {'Technique': tech, 'Avg Score': sum(scores)/len(scores)}
                    for tech, scores in technique_scores.items()
                ])
                st.bar_chart(technique_df.set_index('Technique'))
            
            # Improvement suggestions
            st.subheader("💡 Improvement Insights")
            
            # Analyze low-scoring evaluations
            low_scores = [e for e in st.session_state.evaluation_results if 0 < e.score <= 2]
            if low_scores:
                st.warning(f"Found {len(low_scores)} low-scoring evaluations. Common issues:")
                
                # Extract common feedback themes
                feedbacks = [e.feedback for e in low_scores if e.feedback]
                if feedbacks:
                    for feedback in feedbacks[:3]:
                        st.write(f"• {feedback}")
                
                st.info("Consider adding more examples or refining the prompt structure for these cases.")
        else:
            st.info("Run some evaluations to see analytics!")
        
        # Export functionality
        st.divider()
        st.subheader("📤 Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Prompt History", use_container_width=True):
                # Create JSON export
                history_data = [asdict(v) for v in st.session_state.prompt_history]
                json_data = json.dumps(history_data, indent=2, default=str)
                
                st.download_button(
                    "Download JSON",
                    data=json_data,
                    file_name=f"prompt_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Export Examples", use_container_width=True):
                # Create JSON export
                examples_data = [asdict(e) for e in st.session_state.examples]
                json_data = json.dumps(examples_data, indent=2, default=str)
                
                st.download_button(
                    "Download JSON",
                    data=json_data,
                    file_name=f"examples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("Export Evaluations", use_container_width=True):
                # Create CSV export
                eval_data = []
                for e in st.session_state.evaluation_results:
                    eval_data.append({
                        'timestamp': e.created_at,
                        'test_input': e.test_input,
                        'score': e.score,
                        'feedback': e.feedback
                    })
                
                df = pd.DataFrame(eval_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()