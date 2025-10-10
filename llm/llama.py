import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.debug("started importing llm initiated")

from groq import Groq
import time
from config import GROK_API_KEY
import threading
import hashlib
import json
import os
import re
import uuid
from cachetools import TTLCache
from exploitdb import ExploitDB
import asyncio
import concurrent.futures
import ast
import subprocess
import psutil
from typing import Dict, Any, Optional, List

def is_valid_python_code(code: str) -> bool:
    """Return True if `code` is valid Python syntax. Used only when explicitly requested."""
    logger.debug(f"Checking if code is valid python: {code[:50]}...")
    try:
        ast.parse(code)
        logger.debug("Code is valid Python syntax")
        return True
    except SyntaxError:
        logger.debug("Code is NOT valid Python syntax")
        return False

class ParallelExecutor:
    """
    Async manager for long-running subprocesses started via asyncio.create_subprocess_shell.
    By default we treat inputs as shell commands. If you want to validate Python syntax before
    running Python code, pass check_syntax=True to execute().
    """
    def __init__(self):
        logger.debug("Initializing ParallelExecutor")
        self.executor = concurrent.futures.ThreadPoolExecutor()
        self.running_tasks: Dict[str, Dict[str, Any]] = {}

    async def execute(self, code: str, executor_id: Optional[str] = None, check_syntax: bool = False):
        logger.debug("Starting execute method")
        if executor_id is None:
            executor_id = f"exec_{uuid.uuid4().hex[:8]}"
            logger.debug(f"Generated executor id: {executor_id}")

        logger.debug(f"Executing code with ID: {executor_id}")
        logger.debug(f"Code to execute (preview): {code[:80]}...")

        # Optional python syntax check (useful when running python -c or code strings)
        if check_syntax:
            if not is_valid_python_code(code):
                logger.error("Syntax error detected in provided python code")
                return executor_id, SyntaxError("Invalid Python syntax")

        # Launch as a shell command (use bash -lc to ensure consistent behavior across platforms where bash exists)
        logger.debug("Creating subprocess (asyncio.create_subprocess_shell)")
        process = await asyncio.create_subprocess_shell(
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE
        )
        logger.debug(f"Subprocess created with pid={process.pid}")
        self.running_tasks[executor_id] = {
            "process": process,
            "start_time": time.time(),
            "last_output": ""
        }
        return executor_id, None

    async def get_result(self, executor_id: str) -> Optional[str]:
        logger.debug("Starting get_result method")
        if executor_id in self.running_tasks:
            task = self.running_tasks[executor_id]
            process = task["process"]
            # If process finished, .returncode will be set
            if process.returncode is not None:
                stdout, stderr = await process.communicate()
                del self.running_tasks[executor_id]
                output = stdout.decode() if stdout else stderr.decode()
                logger.debug(f"Process finished, returning output preview: {output[:100]}...")
                return output
            else:
                logger.debug("Process is still running")
                return None
        logger.debug("Executor ID not found")
        return None

    async def stop_execution(self, executor_id: str) -> Optional[str]:
        logger.debug("Starting stop_execution")
        if executor_id in self.running_tasks:
            task = self.running_tasks[executor_id]
            process = task["process"]
            # terminate then collect output
            try:
                process.terminate()
            except ProcessLookupError:
                logger.debug("Process already exited before terminate()")
            await process.wait()
            stdout, stderr = await process.communicate()
            last_output = stdout.decode() if stdout else stderr.decode()
            del self.running_tasks[executor_id]
            logger.debug(f"Process terminated, last output preview: {last_output[:100]}...")
            return last_output
        logger.debug("Executor ID not found")
        return None

    def list_processes(self) -> List[Dict[str, Any]]:
        logger.debug("Starting list_processes method")
        return [
            {
                "id": executor_id,
                "runtime": time.time() - task["start_time"],
                "command": f"Process {getattr(task['process'], 'pid', 'unknown')}"
            }
            for executor_id, task in self.running_tasks.items()
        ]

    async def handle_input(self, executor_id: str, input_data: str) -> bool:
        logger.debug("Starting handle_input")
        if executor_id in self.running_tasks:
            task = self.running_tasks[executor_id]
            process = task["process"]
            if process.stdin:
                process.stdin.write(input_data.encode() + b'\n')
                try:
                    await process.stdin.drain()
                except Exception:
                    # Some platforms/implementations may not support drain; ignore if it fails
                    logger.debug("drain() on stdin failed or not supported")
                logger.debug("Input written to process stdin")
                return True
            else:
                logger.debug("Process has no stdin")
                return False
        logger.debug("Executor ID not found")
        return False

class LLM:
    def __init__(self):
        self.client = Groq(api_key=GROK_API_KEY)
        self.parallel_executor = ParallelExecutor()
        self.exploitdb = ExploitDB()
        self.messages: List[Dict[str, str]] = []  # initialize messages used by chain_of_thought
        logger.debug("LLM initialized")

    async def generate(self, prompt: str) -> Optional[str]:
        logger.debug(f"Generating response for prompt (preview): {prompt[:60]}...")
        try:
            loop = asyncio.get_running_loop()
            # run the blocking client call in a thread pool
            chat_completion = await loop.run_in_executor(None, lambda: self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3-groq-70b-8192-tool-use-preview",
                max_tokens=1024,
            ))
            # guard access to nested attributes
            content = None
            try:
                content = chat_completion.choices[0].message.content
            except Exception:
                # fallback if API shape differs
                try:
                    content = chat_completion["choices"][0]["message"]["content"]
                except Exception:
                    content = str(chat_completion)
            logger.debug(f"LLM response preview: {content[:80] if content else 'None'}")
            return content
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return None

    async def execute_command(self, command: str, check_syntax: bool = False) -> str:
        logger.debug(f"Executing command: {command[:80]}...")
        executor_id, err = await self.parallel_executor.execute(command, check_syntax=check_syntax)
        if err is not None:
            logger.error(f"Error starting execution: {err}")
        return executor_id

    async def get_command_result(self, executor_id: str) -> Optional[str]:
        logger.debug(f"Getting command result for executor ID: {executor_id}")
        result = await self.parallel_executor.get_result(executor_id)
        logger.debug(f"Command result preview: {result[:100] if result else 'None'}")
        return result

    async def stop_command(self, executor_id: str) -> Optional[str]:
        logger.debug(f"Stopping command for executor ID: {executor_id}")
        result = await self.parallel_executor.stop_execution(executor_id)
        logger.debug(f"Command stopped, last output preview: {result[:100] if result else 'None'}")
        return result

    def list_running_processes(self) -> List[Dict[str, Any]]:
        logger.debug("Listing running processes")
        return self.parallel_executor.list_processes()

    async def handle_command_input(self, executor_id: str, input_data: str) -> bool:
        logger.debug(f"Handling input for executor ID: {executor_id}")
        return await self.parallel_executor.handle_input(executor_id, input_data)

    async def chain_of_thought(self, prompt: str, max_iterations: int = 5):
        """
        Async generator that yields (executor_id, code) when code blocks are found.
        Usage:
            async for exec_id, code in llm.chain_of_thought("my prompt"):
                # do something with code / start execution
        """
        logger.debug(f"Starting chain_of_thought with prompt preview: {prompt[:60]}...")
        # ensure messages exists
        if not hasattr(self, "messages") or self.messages is None:
            self.messages = []
        self.messages.append({"role": "user", "content": prompt})

        for i in range(max_iterations):
            iteration_prompt = f"Iteration {i + 1}: {prompt}"
            response = await self.generate(iteration_prompt)
            if response is None:
                logger.debug("Received no response from generate(); stopping")
                break

            self.messages.append({"role": "assistant", "content": response})
            # find python code block
            code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
                executor_id = f"exec_{uuid.uuid4().hex[:8]}"
                logger.debug(f"Yielding code found in iteration {i+1}, preview: {code[:80]}...")
                yield executor_id, code

            if "FINAL ANSWER:" in response:
                logger.debug("FINAL ANSWER found, stopping chain_of_thought")
                break

        # final content is the last assistant message (if any)
        if self.messages and self.messages[-1]["role"] == "assistant":
            return self.messages[-1]["content"]
        return None

    def search_exploits(self, vulnerability: str) -> str:
        logger.debug(f"Searching exploits for vulnerability (preview): {vulnerability[:60]}...")
        exploits = None

        # try common patterns for the exploitdb wrapper
        try:
            if hasattr(self, 'exploitdb') and self.exploitdb is not None:
                # if exploitdb has a search/query method
                if hasattr(self.exploitdb, 'search') and callable(getattr(self.exploitdb, 'search')):
                    exploits = self.exploitdb.search(vulnerability)
                elif hasattr(self.exploitdb, 'query') and callable(getattr(self.exploitdb, 'query')):
                    exploits = self.exploitdb.query(vulnerability)
                elif callable(self.exploitdb):
                    # fallback if the object itself is callable
                    exploits = self.exploitdb(vulnerability)
                else:
                    logger.debug("exploitdb object does not expose 'search'/'query' or is not callable")
        except Exception as e:
            logger.error(f"Error calling exploitdb API: {e}")
            exploits = None

        if exploits:
            # try to build a list safely
            try:
                top = exploits[:5]
                formatted = "\n".join([f"- {e.get('title','unknown')} (ID: {e.get('id','?')})" for e in top])
                return f"Relevant exploits found for {vulnerability}:\n{formatted}"
            except Exception:
                # generic fallback formatting
                try:
                    return f"Relevant exploits found for {vulnerability}:\n{str(exploits)[:1000]}"
                except Exception:
                    return f"Relevant exploits found for {vulnerability} (raw)."
        return f"No specific exploits found for {vulnerability} in the ExploitDB."

    async def analyze_vulnerability(self, vulnerability: str) -> Optional[str]:
        logger.debug(f"Analyzing vulnerability (preview): {vulnerability[:60]}...")
        exploit_info = self.search_exploits(vulnerability)
        analysis_prompt = (
            f"Analyze the following vulnerability and provide insights based on the available exploit information:\n\n"
            f"Vulnerability: {vulnerability}\n\n"
            f"Exploit Information:\n{exploit_info}\n\n"
            "Provide a detailed analysis, including potential impact, exploitation difficulty, and recommended mitigation steps."
        )
        return await self.generate(analysis_prompt)

    async def analyze_vulnerability(self, vulnerability):
        logger.debug(f"Analyzing vulnerability: {vulnerability[:50]}...")
        exploit_info = self.search_exploits(vulnerability)
        analysis_prompt = f"Analyze the following vulnerability and provide insights based on the available exploit information:\n\nVulnerability: {vulnerability}\n\nExploit Information:\n{exploit_info}\n\nProvide a detailed analysis, including potential impact, exploitation difficulty, and recommended mitigation steps."
        return await self.generate(analysis_prompt)

