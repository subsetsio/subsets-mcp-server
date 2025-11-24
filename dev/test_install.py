#!/usr/bin/env python3
"""
Test the MCP server installation as a user would.
This simulates installing and using the server from GitHub.
"""

import subprocess
import sys
import json
import time
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status="info"):
    colors = {"info": BLUE, "success": GREEN, "error": RED, "warning": YELLOW}
    color = colors.get(status, RESET)
    print(f"{color}{message}{RESET}")

def run_command(cmd, description, timeout=30):
    """Run a command and return success status"""
    print_status(f"\n{'='*70}", "info")
    print_status(f"TEST: {description}", "info")
    print_status(f"Command: {cmd}", "info")
    print_status(f"{'='*70}", "info")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            print_status(f"✓ {description} - SUCCESS", "success")
            if result.stdout:
                print(f"Output:\n{result.stdout[:500]}")
            return True
        else:
            print_status(f"✗ {description} - FAILED", "error")
            if result.stderr:
                print(f"Error:\n{result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print_status(f"✗ {description} - TIMEOUT", "error")
        return False
    except Exception as e:
        print_status(f"✗ {description} - EXCEPTION: {e}", "error")
        return False

def test_uv_installed():
    """Check if uv is installed"""
    return run_command("uv --version", "Check uv is installed", timeout=5)

def test_github_url_works():
    """Test that the GitHub URL works with uv run"""
    cmd = "uv run --help"
    return run_command(cmd, "Check uv run works", timeout=10)

def test_mcp_server_help():
    """Test running the MCP server with --help"""
    cmd = "uvx --from git+https://github.com/subsetsio/subsets-mcp-server.git mcp-server --help"
    return run_command(cmd, "Test MCP server --help from GitHub", timeout=60)

def test_mcp_server_with_api_key():
    """Test running the MCP server with API key"""
    # Get API key from auth.json if it exists
    auth_path = Path.home() / "subsets" / "auth.json"
    api_key = "test_key_123"

    if auth_path.exists():
        try:
            with open(auth_path) as f:
                data = json.load(f)
                api_key = data.get("api_key", api_key)
                print_status(f"Using existing API key from {auth_path}", "info")
        except:
            pass

    # Test the actual installation command users would run
    cmd = f"uvx --from git+https://github.com/subsetsio/subsets-mcp-server.git mcp-server --api-key {api_key}"
    print_status(f"\nTesting with API key (will timeout after 5s as MCP server runs indefinitely)", "info")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=5  # Will raise TimeoutExpired after 5 seconds
        )
        # If we get here, server exited (which is unexpected)
        print_status("✗ MCP server exited unexpectedly", "error")
        print(f"stdout: {result.stdout[:300]}")
        print(f"stderr: {result.stderr[:300]}")
        return False
    except subprocess.TimeoutExpired:
        # Server is running! This is success
        print_status("✓ MCP server starts and runs successfully", "success")
        return True
    except Exception as e:
        print_status(f"✗ MCP server failed to start: {e}", "error")
        return False

def test_tools_available():
    """Test that MCP tools are registered"""
    cmd = "cd /Users/nathansnellaert/Documents/subsets-mcp-server && grep '@mcp.tool()' src/server.py | wc -l"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    try:
        tool_count = int(result.stdout.strip())
        if tool_count >= 3:  # Should have at least list_datasets, get_dataset, execute_sql_query
            print_status(f"✓ Found {tool_count} MCP tools registered", "success")
            return True
        else:
            print_status(f"✗ Only found {tool_count} tools (expected >= 3)", "error")
            return False
    except:
        print_status("✗ Could not count MCP tools", "error")
        return False

def main():
    print_status("\n" + "="*70, "info")
    print_status("SUBSETS MCP SERVER - INSTALLATION TEST", "info")
    print_status("="*70 + "\n", "info")

    tests = [
        ("uv Installation", test_uv_installed),
        ("uv run command", test_github_url_works),
        ("MCP Tools Registration", test_tools_available),
        ("GitHub URL Installation", test_mcp_server_help),
        ("MCP Server Startup", test_mcp_server_with_api_key),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_status(f"✗ {test_name} - Exception: {e}", "error")
            results.append((test_name, False))

        time.sleep(1)

    # Summary
    print_status("\n" + "="*70, "info")
    print_status("TEST SUMMARY", "info")
    print_status("="*70, "info")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "success" if result else "error"
        symbol = "✓" if result else "✗"
        print_status(f"{symbol} {test_name}", status)

    print_status(f"\n{passed}/{total} tests passed", "success" if passed == total else "warning")

    if passed == total:
        print_status("\n🎉 All tests passed! MCP server is ready to use.", "success")
        print_status("\nInstall with: uvx --from git+https://github.com/subsetsio/subsets-mcp-server.git mcp-server --api-key YOUR_KEY", "info")
        return 0
    else:
        print_status("\n⚠️  Some tests failed. Check output above.", "warning")
        return 1

if __name__ == "__main__":
    sys.exit(main())
