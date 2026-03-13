# Makes the sandbox/ folder a Python package.
# Exposes both sandbox classes at the top level so callers can write:
#   from sandbox import StdioSandbox, LeetCodeSandbox
# instead of importing from the individual files directly.

from .stdio_sandbox import StdioSandbox        # for APPS + LCB stdin/stdout problems
from .leetcode_sandbox import LeetCodeSandbox  # for LCB LeetCode function-call problems

# explicitly declare what this package exports
__all__ = ["StdioSandbox", "LeetCodeSandbox"]