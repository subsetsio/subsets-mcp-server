import os
import sys
import json
import shutil
import re
import argparse
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from fastmcp import FastMCP
from subsets_platform.local import LocalDataPlatform
from subsets_platform.remote import RemoteDataPlatform
from auth import get_api_key, get_api_url, save_api_key
from sync import sync_dataset, sync_all_datasets, get_batch_metadata

# Create MCP server
mcp = FastMCP("Subsets Data Warehouse")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Subsets MCP Server (Local)")
parser.add_argument("--api-key", type=str, help="Subsets API key for remote catalog access")
args, unknown = parser.parse_known_args()

# Save API key if provided via CLI
if args.api_key:
    save_api_key(args.api_key)

# Get configuration
SUBSETS_API_URL = get_api_url()
SUBSETS_API_KEY = get_api_key()

# Initialize platforms
local_platform = LocalDataPlatform()
remote_platform = RemoteDataPlatform(SUBSETS_API_URL, SUBSETS_API_KEY) if SUBSETS_API_KEY else None


def format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.0f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f} KB"
    else:
        return f"{size_bytes} B"


def calculate_disk_usage(path: Path) -> int:
    """Calculate total disk usage of a directory in bytes."""
    total = 0
    if path.exists():
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
    return total


def extract_table_names(sql: str) -> List[str]:
    """Extract table names from SQL query."""
    # Simple regex to find FROM and JOIN table references
    pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    matches = re.findall(pattern, sql, re.IGNORECASE)
    return list(set(matches))


# =============================================================================
# SHARED TOOLS (Available in both hosted and local mode)
# =============================================================================

@mcp.tool()
def search_datasets(query: str, limit: int = 20) -> Dict[str, Any]:
    """Search the Subsets catalog for datasets.

    Always searches the full remote catalog, regardless of what's installed locally.
    Use this to discover datasets, then use add_dataset() to download them.

    Args:
        query: Search keywords (e.g. "gdp growth", "carbon emissions")
        limit: Maximum results to return (1-100)

    Returns:
        Dictionary with query, results array, and total count.
        Each result includes: dataset_id, title, description, row_count, columns, score
    """
    if not remote_platform:
        return {"error": "API key required. Configure SUBSETS_API_KEY."}

    return remote_platform.search_datasets(query, limit)


@mcp.tool()
def inspect_datasets(dataset_ids: List[str]) -> Dict[str, Any]:
    """Inspect datasets to get detailed information, statistics, and preview.

    Returns metadata, schema, preview rows, column statistics, and usage info.
    Use this to evaluate datasets before downloading or querying.

    Args:
        dataset_ids: List of dataset identifiers (1-20 datasets)

    Returns:
        Dictionary with results for each dataset:
        - Metadata (title, description, license, source info)
        - Schema with all columns and types
        - Statistics (row count, size, column cardinality, min/max/mean for numerics)
        - Preview of first rows
        - Usage info (downloads, frequently_used_with) for catalog datasets
        - Sync status (installed, local_version, last_synced) for local datasets
    """
    if not dataset_ids:
        return {"error": "At least one dataset_id required"}

    if len(dataset_ids) > 20:
        return {"error": "Maximum 20 datasets per request"}

    if not remote_platform:
        return {"error": "API key required. Configure SUBSETS_API_KEY."}

    # Get local dataset info for sync status
    local_datasets = {d['id']: d for d in local_platform.list_local_datasets()}

    results = {}
    for dataset_id in dataset_ids:
        details = remote_platform.get_dataset_details(dataset_id)

        # Add local sync status if installed
        if dataset_id in local_datasets:
            local_info = local_datasets[dataset_id]
            details["installed"] = True
            details["local_version"] = local_info.get("local_version")
            details["last_synced"] = local_info.get("last_synced")
            details["synced"] = local_info.get("synced", False)
        else:
            details["installed"] = False

        results[dataset_id] = details

    return {
        "count": len(results),
        "datasets": results
    }


# =============================================================================
# QUERY TOOL (Local DuckDB)
# =============================================================================

@mcp.tool()
def execute_query(query: str, output_format: str = "tsv") -> Dict[str, Any]:
    """Execute a SQL query against your locally installed datasets.

    Runs on local DuckDB for speed and privacy. Only installed datasets are available.
    Use list_datasets() to see what's available, or add_datasets() to install more.

    Args:
        query: SQL query to execute
        output_format: 'json' or 'tsv' (more efficient for large results)

    Returns:
        Query results in the requested format
    """
    if output_format not in ["json", "tsv"]:
        return {"error": "Invalid output_format. Must be 'json' or 'tsv'"}

    # Check if referenced tables are installed
    table_names = extract_table_names(query)
    installed = {d['id'] for d in local_platform.list_local_datasets() if d.get('synced')}

    missing = []
    for table in table_names:
        # Normalize table name (replace _ with - for matching)
        normalized = table.replace('_', '-')
        if table not in installed and normalized not in installed:
            # Check with underscores too
            underscore_version = table.replace('-', '_')
            if underscore_version not in installed:
                missing.append(table)

    if missing:
        return {
            "error": f"Dataset(s) not installed locally: {', '.join(missing)}",
            "suggestion": f"Use add_dataset('{missing[0]}') to download it first.",
            "missing_datasets": missing
        }

    return local_platform.execute_sql_query(query, output_format=output_format)


# =============================================================================
# LOCAL COLLECTION TOOLS
# =============================================================================

@mcp.tool()
def list_datasets(query: Optional[str] = None) -> Dict[str, Any]:
    """List datasets in your local collection.

    Shows what's available for querying with execute_query().
    Optionally filter by keyword to find specific datasets.

    Args:
        query: Optional keyword to filter datasets by name/id

    Returns:
        Dictionary with:
        - total: Number of matching datasets
        - datasets: List with id, synced status, size, row_count
    """
    datasets = local_platform.list_local_datasets()
    synced = [d for d in datasets if d.get('synced', False)]

    # Filter by query if provided
    if query:
        query_lower = query.lower()
        synced = [d for d in synced if query_lower in d['id'].lower()]

    return {
        "total": len(synced),
        "datasets": [{
            "id": d["id"],
            "row_count": d.get("row_count"),
            "size_bytes": d.get("size_bytes"),
            "size_formatted": format_bytes(d.get("size_bytes", 0) or 0),
            "local_version": d.get("local_version"),
            "last_synced": d.get("last_synced")
        } for d in synced]
    }


@mcp.tool()
def add_datasets(dataset_ids: List[str]) -> Dict[str, Any]:
    """Download datasets to your local collection.

    After downloading, you can query them with execute_query().
    This is a blocking operation that waits for downloads to complete.

    Args:
        dataset_ids: List of dataset identifiers (e.g., ['wdi_gdp_growth', 'wdi_population'])

    Returns:
        Dictionary with download status for each dataset
    """
    if not dataset_ids:
        return {"error": "At least one dataset_id required"}

    if not remote_platform:
        return {
            "success": False,
            "error": "API key required to download datasets. Configure SUBSETS_API_KEY."
        }

    # Fetch metadata for all datasets
    metadata = get_batch_metadata(dataset_ids)
    if not metadata:
        return {
            "success": False,
            "error": "Could not fetch dataset metadata"
        }

    # Check which datasets exist
    not_found = [did for did in dataset_ids if did not in metadata]
    if not_found:
        return {
            "success": False,
            "error": f"Datasets not found: {', '.join(not_found)}"
        }

    # Add datasets to collection
    existing = local_platform.config.get("datasets", [])
    existing_ids = {d['id'] for d in existing}

    datasets_to_add = []
    for dataset_id in dataset_ids:
        if dataset_id not in existing_ids:
            dataset_info = metadata[dataset_id]
            datasets_to_add.append({
                'id': dataset_id,
                'name': dataset_id,
                'added_at': datetime.now().isoformat(),
                'size_bytes': dataset_info.get('size_bytes'),
                'row_count': dataset_info.get('row_count'),
                'delta_version': dataset_info.get('delta_version'),
                'last_modified': dataset_info.get('last_modified')
            })

    if datasets_to_add:
        local_platform.add_datasets(datasets_to_add)

    # Sync all datasets
    data_path = local_platform.data_path
    results = {"added": [], "failed": [], "already_installed": []}
    total_bytes = 0

    for dataset_id in dataset_ids:
        sync_result = sync_dataset(dataset_id, data_path, quiet=True)

        if sync_result["status"] == "failed":
            results["failed"].append({
                "dataset_id": dataset_id,
                "error": sync_result.get("error", "Download failed")
            })
        elif sync_result["status"] == "up_to_date":
            results["already_installed"].append(dataset_id)
        else:
            bytes_downloaded = sync_result.get("bytes_downloaded", 0)
            total_bytes += bytes_downloaded
            results["added"].append({
                "dataset_id": dataset_id,
                "bytes_downloaded": bytes_downloaded
            })

    return {
        "success": len(results["failed"]) == 0,
        "added": len(results["added"]),
        "already_installed": len(results["already_installed"]),
        "failed": len(results["failed"]),
        "total_bytes_downloaded": total_bytes,
        "total_formatted": format_bytes(total_bytes),
        "details": results,
        "message": f"Added {len(results['added'])} dataset(s). Ready to query."
    }


@mcp.tool()
def remove_datasets(dataset_ids: List[str]) -> Dict[str, Any]:
    """Remove datasets from your local collection and delete their data.

    Args:
        dataset_ids: List of dataset identifiers to remove

    Returns:
        Dictionary with removal status and bytes freed
    """
    if not dataset_ids:
        return {"error": "At least one dataset_id required"}

    existing = local_platform.config.get("datasets", [])
    existing_ids = {d['id'] for d in existing}

    not_installed = [did for did in dataset_ids if did not in existing_ids]
    if not_installed:
        return {
            "success": False,
            "error": f"Datasets not installed: {', '.join(not_installed)}"
        }

    total_bytes_freed = 0
    removed = []

    for dataset_id in dataset_ids:
        # Calculate size before deletion
        dataset_path = local_platform.data_path / dataset_id
        bytes_freed = calculate_disk_usage(dataset_path)
        total_bytes_freed += bytes_freed

        # Delete local data
        if dataset_path.exists():
            shutil.rmtree(dataset_path)

        removed.append({"dataset_id": dataset_id, "bytes_freed": bytes_freed})

    # Remove from collection
    local_platform.remove_datasets(dataset_ids)

    return {
        "success": True,
        "removed": len(removed),
        "total_bytes_freed": total_bytes_freed,
        "total_formatted": format_bytes(total_bytes_freed),
        "details": removed,
        "message": f"Removed {len(removed)} dataset(s), freed {format_bytes(total_bytes_freed)}"
    }


@mcp.tool()
def sync_datasets(dataset_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """Update installed datasets to their latest versions.

    Downloads only the changes (Delta Lake diffs) for efficiency.
    If no IDs specified, syncs all installed datasets.

    Args:
        dataset_ids: Optional list of dataset IDs to sync. If None, syncs all.

    Returns:
        Dictionary with sync results
    """
    if not remote_platform:
        return {
            "success": False,
            "error": "API key required to sync. Configure SUBSETS_API_KEY."
        }

    all_datasets = local_platform.list_local_datasets()

    if not all_datasets:
        return {
            "success": True,
            "message": "No datasets installed. Use add_datasets() first.",
            "synced": 0,
            "failed": 0,
            "up_to_date": 0
        }

    if dataset_ids:
        datasets_to_sync = [d for d in all_datasets if d['id'] in dataset_ids]
        not_found = set(dataset_ids) - {d['id'] for d in datasets_to_sync}
        if not_found:
            return {
                "success": False,
                "error": f"Datasets not installed: {', '.join(not_found)}"
            }
    else:
        datasets_to_sync = all_datasets

    data_path = local_platform.data_path
    results = sync_all_datasets(datasets_to_sync, data_path, max_workers=4)

    total_bytes = sum(r.get("bytes_downloaded", 0) for r in results["synced"])

    return {
        "success": True,
        "synced": len(results["synced"]),
        "up_to_date": len(results["up_to_date"]),
        "failed": len(results["failed"]),
        "total_bytes_downloaded": total_bytes,
        "total_formatted": format_bytes(total_bytes),
        "details": {
            "synced": [r["dataset_id"] for r in results["synced"]],
            "up_to_date": [r["dataset_id"] for r in results["up_to_date"]],
            "failed": [{
                "dataset_id": r["dataset_id"],
                "error": r.get("error")
            } for r in results["failed"]]
        }
    }


def main():
    """Main entry point for the MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
