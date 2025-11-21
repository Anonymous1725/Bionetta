"""
Module for pretty printing the details of certain actions in the code.
"""

from typing import Dict, List, Any
import subprocess
import re

from rich.panel import Panel

from tf_bionetta.logging.logger import console


PANIC_PATTERS = [
    r"\bpanicked at\b",
    r"called `Result::unwrap\(\)` on an `Err` value",
    r"thread '[^']*' panicked",
]


def print_success_message(msg: str) -> None:
    """
    Prints the success message in green color
    """
    
    console.print(f"[bold green]{msg}[/bold green] :tada:")


def run_command(
    command: List[str],
    env: Dict[str, str] | None=None,
    output_logs: bool = True,
    *args, **kwargs
) -> Any:
    """
    Runs the specified command and prints the output or error message in a formatted way.
    
    Args:
        command (List[str]): The command to run.
        *args: Positional arguments to pass to the command.
        **kwargs: Keyword arguments to pass to the command.
    """
    
    command_str = " ".join(command)
    console.print(f"[yellow]$ {command_str}[/]")
    
    try:
        result = subprocess.run(command, check=True, text=True, env=env, capture_output=True, *args, **kwargs)

        if output_logs and result.stdout:
            console.print(Panel.fit(result.stdout.strip(), title="Stdout", border_style="green"))

        if any(re.search(p, result.stderr) for p in PANIC_PATTERS):
            console.print(Panel.fit(result.stderr.strip(), title="Panic detected", border_style="red"))
            raise RuntimeError("Rust panic detected in output")
        
        return result

    except subprocess.CalledProcessError as e:
        if e.stdout:
            console.print(Panel.fit(e.stdout, title="Stdout", border_style="yellow"))
        if e.stderr:
            console.print(Panel.fit(e.stderr, title="Stderr", border_style="red"))
        exit()
    
    except RuntimeError as e:
        console.print(Panel.fit(e, title="Stderr", border_style="red"))