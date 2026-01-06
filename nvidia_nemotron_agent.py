"""
External Agent for vLLM-served models via local vLLM server

This agent integrates a locally running vLLM server with Harbor framework
by implementing the BaseAgent interface. It works with any model served via vLLM.
"""

import os
import json
from typing import Optional
from openai import AsyncOpenAI

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class VLLMAgent(BaseAgent):
    """
    External agent that uses any model served via local vLLM server
    with OpenAI-compatible API.
    """

    def __init__(
        self,
        logs_dir,
        model_name: str = None,
        logger = None,
        api_base: str = "http://localhost:8000/v1",
        temperature: float = 0.1,
        max_tokens: int = 8192,
        **kwargs,
    ):
        """
        Initialize the vLLM agent.

        Args:
            logs_dir: Directory for agent logs (required by BaseAgent)
            model_name: Model identifier (must match model served by vLLM)
            logger: Logger instance (optional)
            api_base: Base URL for the vLLM OpenAI-compatible API
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate
        """
        # Call parent constructor
        super().__init__(logs_dir=logs_dir, model_name=model_name, logger=logger, **kwargs)

        # Set model to use - must be provided
        if not model_name:
            raise ValueError("model_name is required and must match the model served by vLLM")
        self.model = model_name
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI client pointing to local vLLM server
        self.client = AsyncOpenAI(
            api_key="dummy-key-for-local-server",  # vLLM doesn't require real API key
            base_url=api_base,
        )

    @staticmethod
    def name() -> str:
        """Return the agent's name."""
        return "vllm-agent"

    @staticmethod
    def version() -> Optional[str]:
        """Return the agent's version."""
        return "2.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """
        Setup the agent and its tools before execution.

        Args:
            environment: The execution environment
        """
        # Verify the vLLM server is accessible
        try:
            # Test connection by listing models
            models = await self.client.models.list()
            print(f"Connected to vLLM server at {self.api_base}")
            print(f"Available models: {[model.id for model in models.data]}")
        except Exception as e:
            print(f"Warning: Could not connect to vLLM server at {self.api_base}: {e}")
            print("Please ensure the vLLM server is running before executing tasks.")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """
        Execute the agent with the given instruction using iterative problem-solving.

        Args:
            instruction: The task to complete
            environment: The execution context
            context: Storage for results
        """
        # System prompt to guide iterative problem-solving
        system_prompt = """You are an autonomous agent tasked with completing challenges in a Linux environment.

Your approach:
1. READ the task carefully and understand what's required
2. PLAN your approach step-by-step
3. EXECUTE commands using bash code blocks (```bash)
4. VERIFY results by checking outputs
5. ITERATE if needed - fix errors, try different approaches
6. When task is complete, say "TASK COMPLETE" clearly

You have access to a Linux environment with bash. To run commands, wrap them in markdown code blocks:
```bash
your command here
```

After each command execution, you'll receive the output. Use this feedback to:
- Check if your action succeeded
- Debug any errors
- Decide your next step

Work iteratively until the task is fully complete. Be thorough and verify your work."""

        # Maximum iterations for task completion
        max_iterations = 15

        # Track conversation history with system prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]

        iteration_count = 0
        task_complete = False

        print(f"\n{'='*60}")
        print(f"STARTING TASK: {instruction[:100]}...")
        print(f"{'='*60}\n")

        # Iterative problem-solving loop
        for iteration in range(max_iterations):
            iteration_count += 1
            print(f"\n--- Iteration {iteration_count}/{max_iterations} ---")

            # Get response from the model
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                assistant_message = response.choices[0].message.content
                print(f"\nAssistant response ({len(assistant_message)} chars):")
                print(f"{assistant_message[:300]}...")
                if len(assistant_message) > 300:
                    print(f"... [{len(assistant_message) - 300} more characters]")

                # Add assistant response to history
                messages.append({
                    "role": "assistant",
                    "content": assistant_message
                })

                # Check for task completion signal
                if "TASK COMPLETE" in assistant_message.upper():
                    print("\n✓ Agent signaled task completion")
                    task_complete = True
                    break

                # Extract and execute bash commands
                commands = self._extract_bash_commands(assistant_message)

                if not commands:
                    # No commands to execute - ask agent to continue or confirm completion
                    print("\nNo commands found. Asking agent to proceed...")
                    messages.append({
                        "role": "user",
                        "content": "No bash commands detected. If the task is complete, say 'TASK COMPLETE'. Otherwise, provide the next bash command to execute."
                    })
                    continue

                # Execute each command and collect feedback
                all_outputs = []
                for cmd_idx, cmd in enumerate(commands, 1):
                    print(f"\n[Command {cmd_idx}/{len(commands)}]")
                    print(f"Executing: {cmd[:100]}...")

                    result = await environment.exec(cmd)

                    # Build execution feedback
                    exec_feedback = f"Command {cmd_idx}: {cmd}\n"
                    exec_feedback += f"Exit code: {result.return_code}\n"

                    if result.stdout:
                        stdout_preview = result.stdout[:1000]
                        exec_feedback += f"STDOUT:\n{stdout_preview}"
                        if len(result.stdout) > 1000:
                            exec_feedback += f"\n... [truncated {len(result.stdout) - 1000} chars]"
                        exec_feedback += "\n"
                        print(f"✓ Exit {result.return_code} | stdout: {len(result.stdout)} chars")

                    if result.stderr:
                        stderr_preview = result.stderr[:1000]
                        exec_feedback += f"STDERR:\n{stderr_preview}"
                        if len(result.stderr) > 1000:
                            exec_feedback += f"\n... [truncated {len(result.stderr) - 1000} chars]"
                        exec_feedback += "\n"
                        print(f"✗ Exit {result.return_code} | stderr: {len(result.stderr)} chars")

                    all_outputs.append(exec_feedback)

                # Send all command results back to the model for reflection
                combined_feedback = "\n---\n".join(all_outputs)
                combined_feedback += "\n\nBased on these results, what should you do next? If the task is complete, say 'TASK COMPLETE'."

                messages.append({
                    "role": "user",
                    "content": combined_feedback
                })

            except Exception as e:
                print(f"\n✗ Error in iteration {iteration_count}: {e}")
                # Add error to conversation so agent can adapt
                messages.append({
                    "role": "user",
                    "content": f"An error occurred: {str(e)}\nPlease try a different approach."
                })

        # Final status
        if task_complete:
            status = "completed_with_signal"
        elif iteration_count >= max_iterations:
            status = "max_iterations_reached"
            print(f"\n⚠ Reached maximum iterations ({max_iterations})")
        else:
            status = "completed"

        print(f"\n{'='*60}")
        print(f"TASK FINISHED: {status} after {iteration_count} iterations")
        print(f"{'='*60}\n")

        # Store final result in context
        if context.metadata is None:
            context.metadata = {}
        context.metadata["iterations"] = iteration_count
        context.metadata["conversation_length"] = len(messages)
        context.metadata["status"] = status
        context.metadata["task_complete_signal"] = task_complete

    def _extract_bash_commands(self, text: str) -> list[str]:
        """
        Extract bash commands from markdown code blocks.

        Args:
            text: Text potentially containing ```bash or ```sh code blocks

        Returns:
            List of bash commands to execute
        """
        commands = []
        lines = text.split('\n')
        in_code_block = False
        current_block = []

        for line in lines:
            if line.strip().startswith('```bash') or line.strip().startswith('```sh'):
                in_code_block = True
                current_block = []
            elif line.strip() == '```' and in_code_block:
                in_code_block = False
                if current_block:
                    # Join multi-line commands
                    commands.append('\n'.join(current_block))
                    current_block = []
            elif in_code_block:
                current_block.append(line)

        return commands


# Export the agent class
__all__ = ['VLLMAgent']

# Backwards compatibility alias
NvidiaNemotronAgent = VLLMAgent
