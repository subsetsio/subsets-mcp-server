import os
import sys
import json
import argparse
from typing import Optional, Dict, Any
from pathlib import Path
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from subsets_platform.local import LocalDataPlatform
from subsets_platform.remote import RemoteDataPlatform
from auth import get_api_key, get_api_url, save_api_key, clear_api_key

# Create an MCP server
mcp = FastMCP("Subsets Data Warehouse")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Subsets MCP Server")
parser.add_argument("--api-key", type=str, help="Subsets API key for remote mode")
args, unknown = parser.parse_known_args()

# If --api-key is provided via CLI, save it
if args.api_key:
    save_api_key(args.api_key)

# Get configuration
SUBSETS_API_URL = get_api_url()
SUBSETS_API_KEY = get_api_key()

# Load or create mode configuration
config_path = Path.home() / "subsets" / "config.json"
config_path.parent.mkdir(exist_ok=True)

if config_path.exists():
    with open(config_path, 'r') as f:
        config = json.load(f)
else:
    config = {"mode": "remote" if SUBSETS_API_KEY else "local"}
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

# Current mode
current_mode = config.get("mode", "remote")

# Initialize platforms
local_platform = LocalDataPlatform()
remote_platform = RemoteDataPlatform(SUBSETS_API_URL, SUBSETS_API_KEY) if SUBSETS_API_KEY else None


# Models for tool inputs
class ListDatasetsInput(BaseModel):
    """Input parameters for listing datasets"""
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of datasets to return")
    offset: int = Field(default=0, ge=0, description="Number of datasets to skip for pagination")
    license: Optional[str] = Field(None, description="Filter by license type")
    git_user: Optional[str] = Field(None, description="Filter by GitHub username (dataset publisher)")
    q: Optional[str] = Field(None, description="Search query for keyword search")
    min_score: Optional[float] = Field(None, ge=0.0, le=2.0, description="Minimum relevance score threshold")
    detailed: bool = Field(default=False, description="Include detailed summary data")


class ExecuteSQLQueryInput(BaseModel):
    """Input parameters for executing SQL query"""
    query: str = Field(..., min_length=1, description="SQL query to execute")


@mcp.tool()
def list_datasets(
    limit: int = 10,
    offset: int = 0,
    license: Optional[str] = None,
    git_user: Optional[str] = None,
    q: Optional[str] = None,
    min_score: Optional[float] = None,
    detailed: bool = False
) -> Dict[str, Any]:
    """List available datasets from the Subsets data warehouse

    Args:
        limit: Maximum number of datasets to return (1-100)
        offset: Number of datasets to skip for pagination
        license: Filter by license type
        git_user: Filter by GitHub username (dataset publisher)
        q: Search query for keyword search
        min_score: Minimum relevance score threshold (0.0-2.0)
        detailed: Include detailed summary data (column stats, query metrics, etc.)

    Returns:
        Dictionary containing datasets list and total count
    """
    # Check mode and use appropriate platform
    if current_mode == "local":
        return local_platform.list_datasets(
            limit=limit,
            offset=offset,
            license=license,
            git_user=git_user,
            q=q,
            min_score=min_score,
            detailed=detailed
        )
    else:
        if not remote_platform:
            return {
                "error": "Remote mode requires API key. Run 'subsets init' to configure."
            }
        return remote_platform.list_datasets(
            limit=limit,
            offset=offset,
            license=license,
            git_user=git_user,
            q=q,
            min_score=min_score,
            detailed=detailed
        )


@mcp.tool()
def execute_sql_query(query: str, output_format: str = "tsv") -> Dict[str, Any]:
    """Execute a SQL query against datasets in the Subsets data warehouse
    
    Args:
        query: SQL query to execute
        output_format: Output format - 'json' or 'tsv' (more efficient for large results)
    
    Returns:
        Query results in the requested format
    """
    # Validate output format
    if output_format not in ["json", "tsv"]:
        return {
            "error": "Invalid output_format. Must be 'json' or 'tsv'"
        }
    
    # Check mode and use appropriate platform
    if current_mode == "local":
        return local_platform.execute_sql_query(query, output_format=output_format)
    else:
        if not remote_platform:
            return {
                "error": "Remote mode requires API key. Run 'subsets init' to configure."
            }
        return remote_platform.execute_sql_query(query, output_format=output_format)


@mcp.tool()
def get_current_mode() -> Dict[str, Any]:
    """Get the current mode (local or remote)"""
    local_datasets = len(local_platform.config.get("datasets", []))
    return {
        "mode": current_mode,
        "local_datasets": local_datasets,
        "api_configured": remote_platform is not None
    }


@mcp.tool()
def switch_mode(mode: str) -> Dict[str, Any]:
    """Switch between local and remote mode
    
    Args:
        mode: Target mode ('local' or 'remote')
    
    Returns:
        Dictionary with status and new mode information
    """
    global current_mode
    
    if mode not in ["local", "remote"]:
        return {
            "error": "Invalid mode. Must be 'local' or 'remote'",
            "current_mode": current_mode
        }
    
    # Check if switching to remote mode is possible
    if mode == "remote" and not remote_platform:
        return {
            "error": "Cannot switch to remote mode: API key not configured",
            "current_mode": current_mode,
            "suggestion": "Run 'subsets init' to configure your API key, then restart the server"
        }
    
    # Get previous mode
    previous_mode = current_mode
    
    # Switch mode
    try:
        current_mode = mode
        
        # Save to config
        config["mode"] = mode
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        local_datasets = len(local_platform.config.get("datasets", []))
        
        return {
            "success": True,
            "previous_mode": previous_mode,
            "current_mode": mode,
            "message": f"Successfully switched from {previous_mode} to {mode} mode",
            "local_datasets": local_datasets if mode == "local" else None,
            "api_configured": remote_platform is not None
        }
    except Exception as e:
        return {
            "error": f"Failed to switch mode: {str(e)}",
            "current_mode": current_mode
        }


@mcp.tool()
def login(api_key: str) -> Dict[str, Any]:
    """Authenticate with your Subsets API key.

    This saves your API key locally and enables remote mode features like
    executing SQL queries against the full data warehouse.

    Get your API key at https://subsets.io/settings

    Args:
        api_key: Your Subsets API key (starts with sk_)

    Returns:
        Dictionary with login status and user info
    """
    global remote_platform, current_mode, config

    if not api_key or not api_key.strip():
        return {
            "success": False,
            "error": "API key is required. Get your key at https://subsets.io/settings"
        }

    api_key = api_key.strip()

    # Validate the API key by making a test request
    try:
        import requests
        response = requests.get(
            f"{SUBSETS_API_URL}/api/users/profile",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )

        if response.status_code == 401:
            return {
                "success": False,
                "error": "Invalid API key. Check your key at https://subsets.io/settings"
            }

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Failed to validate API key: {response.status_code}"
            }

        user_data = response.json()

        # Save the API key
        save_api_key(api_key)

        # Reinitialize remote platform
        remote_platform = RemoteDataPlatform(SUBSETS_API_URL, api_key)

        # Switch to remote mode
        current_mode = "remote"
        config["mode"] = "remote"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        return {
            "success": True,
            "message": f"Logged in as {user_data.get('email', 'user')}",
            "user": {
                "email": user_data.get("email"),
                "username": user_data.get("username")
            },
            "mode": "remote"
        }

    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": f"Failed to connect to Subsets API: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Login failed: {str(e)}"
        }


@mcp.tool()
def logout() -> Dict[str, Any]:
    """Log out and clear your API key.

    This removes your saved API key and switches to local mode.
    You can still query locally synced datasets after logging out.

    Returns:
        Dictionary with logout status
    """
    global remote_platform, current_mode, config

    # Check if already logged out
    if not get_api_key():
        return {
            "success": True,
            "message": "Already logged out",
            "mode": current_mode
        }

    try:
        # Clear the API key
        clear_api_key()

        # Clear remote platform
        remote_platform = None

        # Switch to local mode
        current_mode = "local"
        config["mode"] = "local"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        local_datasets = len(local_platform.config.get("datasets", []))

        return {
            "success": True,
            "message": "Logged out successfully. Switched to local mode.",
            "mode": "local",
            "local_datasets": local_datasets
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Logout failed: {str(e)}"
        }


@mcp.tool()
def get_dataset_details(dataset_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific dataset including schema, summary statistics, and preview

    Args:
        dataset_id: Dataset identifier (required)

    Returns:
        Dictionary with dataset details including:
        - Metadata (title, description, license, source info)
        - Schema with all columns and types
        - Statistics (row count, size, query usage)
        - Preview of first 20 rows
        - Related tables often queried together
    """
    if current_mode == "local":
        return {
            "error": "get_dataset_details is only available in remote mode. Switch to remote mode with switch_mode('remote')."
        }

    if not remote_platform:
        return {
            "error": "Remote mode requires API key. Run 'subsets init' to configure."
        }

    return remote_platform.get_dataset_details(dataset_id)


@mcp.tool()
def search_datasets(query: str, limit: int = 20) -> Dict[str, Any]:
    """Fast keyword search to find relevant datasets.

    Use this for quick dataset discovery. Returns enough context for AI reranking:
    - dataset_id: Unique identifier
    - title: Human-readable name
    - description: Truncated to ~150 chars
    - row_count: Number of rows
    - columns: All column names
    - score: Keyword match relevance (0.0-1.0)

    Works in both local and remote modes using the same pre-built search index.

    Args:
        query: Search keywords (e.g. "gdp growth", "unemployment rate")
        limit: Maximum results to return (1-100)

    Returns:
        Dictionary with query, results array, and total count
    """
    if current_mode == "local":
        return local_platform.search_datasets(query, limit)

    if not remote_platform:
        return {
            "error": "Remote mode requires API key. Run 'subsets init' to configure."
        }

    return remote_platform.search_datasets(query, limit)


@mcp.tool()
def get_local_collection() -> Dict[str, Any]:
    """Get information about your local dataset collection"""
    datasets = local_platform.list_local_datasets()
    synced_count = sum(1 for d in datasets if d.get('synced', False))
    
    return {
        "total_datasets": len(datasets),
        "synced_datasets": synced_count,
        "pending_sync": len(datasets) - synced_count,
        "datasets": datasets
    }


def main():
    """Main entry point for the MCP server"""
    mcp.run()


# Run the server
if __name__ == "__main__":
    main()