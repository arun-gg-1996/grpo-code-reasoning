"""
Smoke test for StdioSandbox and LeetCodeSandbox.
Run this before touching any real dataset.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sandbox import StdioSandbox, LeetCodeSandbox


# ─────────────────────────────────────────
# StdioSandbox tests
# ─────────────────────────────────────────

def test_stdio_correct():
    sandbox = StdioSandbox()
    code = """
n = int(input())
print(n * 2)
"""
    test_cases = [
        {"input": "3", "output": "6"},
        {"input": "5", "output": "10"},
        {"input": "0", "output": "0"},
    ]
    score = sandbox.execute(code, test_cases)
    assert score == 1.0, f"Expected 1.0 got {score}"
    print("✓ stdio correct solution → 1.0")


def test_stdio_partial():
    sandbox = StdioSandbox()
    code = """
n = int(input())
print(n * 2)  # wrong — should be n*3
"""
    test_cases = [
        {"input": "3", "output": "9"},  # fails
        {"input": "0", "output": "0"},  # passes (0*2 == 0*3)
        {"input": "5", "output": "15"},  # fails
    ]
    score = sandbox.execute(code, test_cases)
    assert abs(score - 1 / 3) < 0.01, f"Expected 0.333 got {score}"
    print(f"✓ stdio partial credit → {score:.3f}")


def test_stdio_timeout():
    sandbox = StdioSandbox(timeout=2)
    code = """
while True:
    pass
"""
    test_cases = [{"input": "", "output": ""}]
    score = sandbox.execute(code, test_cases)
    assert score == 0.0, f"Expected 0.0 got {score}"
    print("✓ stdio timeout → 0.0")


def test_stdio_wrong():
    sandbox = StdioSandbox()
    code = """
print("wrong answer")
"""
    test_cases = [{"input": "5", "output": "10"}]
    score = sandbox.execute(code, test_cases)
    assert score == 0.0, f"Expected 0.0 got {score}"
    print("✓ stdio wrong answer → 0.0")


def test_stdio_syntax_error():
    sandbox = StdioSandbox()
    code = """
def broken(
"""
    test_cases = [{"input": "5", "output": "10"}]
    score = sandbox.execute(code, test_cases)
    assert score == 0.0, f"Expected 0.0 got {score}"
    print("✓ stdio syntax error → 0.0")


def test_stdio_whitespace_tolerance():
    sandbox = StdioSandbox()
    code = """
print("  hello  ")
"""
    test_cases = [{"input": "", "output": "hello"}]
    score = sandbox.execute(code, test_cases)
    assert score == 1.0, f"Expected 1.0 got {score}"
    print("✓ stdio whitespace stripping → 1.0")


def test_stdio_max_tests_cap():
    sandbox = StdioSandbox()
    code = "n = int(input()); print(n)"
    test_cases = [{"input": str(i), "output": str(i)} for i in range(20)]
    score = sandbox.execute(code, test_cases, max_tests=10)
    assert score == 1.0, f"Expected 1.0 got {score}"
    print("✓ stdio max_tests cap → 1.0")


# ─────────────────────────────────────────
# LeetCodeSandbox tests
# ─────────────────────────────────────────

def test_lc_single_arg():
    sandbox = LeetCodeSandbox()
    code = """
class Solution:
    def countSeniors(self, details):
        count = 0
        for d in details:
            age = int(d[11:13])
            if age > 60:
                count += 1
        return count
"""
    test_cases = [
        {
            "input": '["7868190130M7522", "5303914400F9211", "9273338290F4010"]',
            "output": "2"
        },
        {
            "input": '["1313579440F2036", "2921522980M5644"]',
            "output": "0"
        },
    ]
    score = sandbox.execute(code, "countSeniors", test_cases)
    assert score == 1.0, f"Expected 1.0 got {score}"
    print("✓ lc single arg → 1.0")


def test_lc_multi_arg():
    sandbox = LeetCodeSandbox()
    # correct solution: for each index i, OR everything except nums[i],
    # then OR with nums[i] shifted left k times
    code = """
class Solution:
    def maximumOr(self, nums, k):
        n = len(nums)
        prefix = [0] * (n + 1)
        suffix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] | nums[i]
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] | nums[i]
        best = 0
        for i in range(n):
            candidate = prefix[i] | (nums[i] << k) | suffix[i + 1]
            best = max(best, candidate)
        return best
"""
    test_cases = [
        {
            "input": "[12, 9]\n1",
            "output": "30"
        },
    ]
    score = sandbox.execute(code, "maximumOr", test_cases)
    assert score == 1.0, f"Expected 1.0 got {score}"
    print("✓ lc multi arg → 1.0")


def test_lc_wrong_answer():
    sandbox = LeetCodeSandbox()
    code = """
class Solution:
    def countSeniors(self, details):
        return 999
"""
    test_cases = [
        {
            "input": '["7868190130M7522", "5303914400F9211", "9273338290F4010"]',
            "output": "2"
        },
    ]
    score = sandbox.execute(code, "countSeniors", test_cases)
    assert score == 0.0, f"Expected 0.0 got {score}"
    print("✓ lc wrong answer → 0.0")


def test_lc_timeout():
    sandbox = LeetCodeSandbox(timeout=2)
    code = """
class Solution:
    def countSeniors(self, details):
        while True:
            pass
"""
    test_cases = [
        {
            "input": '["7868190130M7522"]',
            "output": "1"
        },
    ]
    score = sandbox.execute(code, "countSeniors", test_cases)
    assert score == 0.0, f"Expected 0.0 got {score}"
    print("✓ lc timeout → 0.0")


def test_lc_returns_list():
    sandbox = LeetCodeSandbox()
    code = """
class Solution:
    def twoSum(self, nums, target):
        seen = {}
        for i, n in enumerate(nums):
            if target - n in seen:
                return [seen[target - n], i]
            seen[n] = i
"""
    test_cases = [
        {"input": "[2, 7, 11, 15]\n9", "output": "[0, 1]"},
        {"input": "[3, 2, 4]\n6", "output": "[1, 2]"},
    ]
    score = sandbox.execute(code, "twoSum", test_cases)
    assert score == 1.0, f"Expected 1.0 got {score}"
    print("✓ lc returns list → 1.0")


# ─────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== StdioSandbox ===")
    test_stdio_correct()
    test_stdio_partial()
    test_stdio_timeout()
    test_stdio_wrong()
    test_stdio_syntax_error()
    test_stdio_whitespace_tolerance()
    test_stdio_max_tests_cap()

    print("\n=== LeetCodeSandbox ===")
    test_lc_single_arg()
    test_lc_multi_arg()
    test_lc_wrong_answer()
    test_lc_timeout()
    test_lc_returns_list()

    print("\n✓ All sandbox tests passed")
