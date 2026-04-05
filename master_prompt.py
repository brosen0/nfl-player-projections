import importlib
import os
import subprocess
import asyncio
from datetime import datetime
from prompts_0 import prompts  # from your installed prompts_0 package

# Cached prompt and version info
_PROMPT_CACHE = prompts.MASTER_PROMPT
_LAST_UPDATE = datetime.utcnow()

def _update_prompts_0_repo_sync():
    """
    Synchronous git pull to update prompts_0 package and reload module.
    Updates cache and timestamp.
    """
    global _PROMPT_CACHE, _LAST_UPDATE
    try:
        import prompts_0
        pkg_path = os.path.dirname(prompts_0.__file__)
        subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=pkg_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
        importlib.reload(prompts)
        _PROMPT_CACHE = prompts.MASTER_PROMPT
        _LAST_UPDATE = datetime.utcnow()
        print(f"[master_prompt] Updated prompt cache at {_LAST_UPDATE.isoformat()}")
    except Exception as e:
        print(f"[master_prompt] Warning: Could not update prompts_0: {e}")

async def _update_prompts_0_repo_async():
    """
    Asynchronous git pull for non-blocking update.
    Updates cache and timestamp.
    """
    global _PROMPT_CACHE, _LAST_UPDATE
    try:
        import prompts_0
        pkg_path = os.path.dirname(prompts_0.__file__)
        proc = await asyncio.create_subprocess_exec(
            "git", "pull", "origin", "main",
            cwd=pkg_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        await proc.communicate()
        importlib.reload(prompts)
        _PROMPT_CACHE = prompts.MASTER_PROMPT
        _LAST_UPDATE = datetime.utcnow()
        print(f"[master_prompt] Async updated prompt cache at {_LAST_UPDATE.isoformat()}")
    except Exception as e:
        print(f"[master_prompt] Warning: Could not async-update prompts_0: {e}")

def get_master_prompt(async_update: bool = True) -> str:
    """
    Returns the master prompt string immediately.

    Parameters:
    - async_update (bool): 
        True (default) → schedule async refresh in background if possible
        False → only return cached prompt

    Returns:
    - str: cached master prompt
    """
    if async_update:
        try:
            asyncio.create_task(_update_prompts_0_repo_async())
        except RuntimeError:
            # No running event loop; fallback to sync update
            _update_prompts_0_repo_sync()
    return _PROMPT_CACHE

def get_prompt_version() -> str:
    """
    Returns a string describing when the prompt cache was last updated.
    Useful for logging and auditability.
    """
    return _LAST_UPDATE.isoformat()




prompt = get_master_prompt()  # Immediate access, background update
version = get_prompt_version()
print(f"Using prompt from {version}")

