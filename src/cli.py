#!/usr/bin/env python3
"""Subsets CLI - Command line interface for the Subsets data platform"""
import typer
import json
import shutil
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import sqlite3

# Initialize Typer app
app = typer.Typer(help="Subsets CLI - Local-first data platform")


def get_local_platform():
    """Get or create the local platform instance"""
    from subsets_platform.local import LocalDataPlatform
    if not hasattr(get_local_platform, '_instance'):
        get_local_platform._instance = LocalDataPlatform()
    return get_local_platform._instance


def get_remote_platform():
    """Get or create the remote platform instance"""
    from subsets_platform.remote import RemoteDataPlatform
    from auth import get_api_key, get_api_url
    if not hasattr(get_remote_platform, '_instance'):
        api_url = get_api_url()
        api_key = get_api_key()
        get_remote_platform._instance = RemoteDataPlatform(api_url, api_key) if api_key else None
    return get_remote_platform._instance


def format_bytes(size_bytes: int) -> str:
    """Format bytes as human-readable string"""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.2f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.0f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.0f} KB"
    else:
        return f"{size_bytes} B"


# =============================================================================
# AUTH COMMANDS
# =============================================================================

@app.command()
def login():
    """Log in with your Subsets API key"""
    from auth import save_api_key, get_api_key

    existing_key = get_api_key()
    if existing_key:
        typer.echo(f"Current API key: {existing_key[:7]}...")
        if not typer.confirm("Update your API key?"):
            return

    api_key = typer.prompt("Enter your Subsets API key", hide_input=True)

    if not api_key.startswith("sk_"):
        typer.echo("Warning: API keys usually start with 'sk_'")
        if not typer.confirm("Continue anyway?"):
            return

    save_api_key(api_key)
    typer.echo("Logged in successfully.")


@app.command()
def logout():
    """Log out and clear your API key"""
    from auth import clear_api_key, get_api_key

    if not get_api_key():
        typer.echo("Already logged out")
        return

    clear_api_key()
    typer.echo("Logged out.")


# =============================================================================
# DISCOVERY COMMANDS (always remote)
# =============================================================================

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum results")
):
    """Search the Subsets catalog for datasets"""
    remote = get_remote_platform()
    if not remote:
        typer.echo("Error: Requires API key. Run 'subsets login'")
        raise typer.Exit(1)

    results = remote.search_datasets(query, limit)

    if "error" in results:
        typer.echo(f"Error: {results['error']}")
        raise typer.Exit(1)

    datasets = results.get("results", [])

    if not datasets:
        typer.echo(f"No datasets found for '{query}'")
        return

    typer.echo(f"Found {len(datasets)} dataset(s):\n")

    for d in datasets:
        dataset_id = d.get('dataset_id', '')
        title = d.get('title', '')[:50]
        row_count = d.get('row_count', 0)

        typer.echo(f"  {dataset_id}")
        if title and title != dataset_id:
            typer.echo(f"    {title}")
        if row_count:
            typer.echo(f"    {row_count:,} rows")
        typer.echo("")

    typer.echo("Use 'subsets add <dataset_id>' to download a dataset.")


@app.command()
def inspect(dataset_ids: List[str] = typer.Argument(..., help="Dataset ID(s) to inspect")):
    """Inspect datasets - get detailed info, statistics, and preview"""
    if len(dataset_ids) > 20:
        typer.echo("Error: Maximum 20 datasets per request")
        raise typer.Exit(1)

    remote = get_remote_platform()
    if not remote:
        typer.echo("Error: Requires API key. Run 'subsets login'")
        raise typer.Exit(1)

    # Get local datasets for sync status
    local_datasets = {d['id']: d for d in get_local_platform().list_local_datasets()}

    for dataset_id in dataset_ids:
        result = remote.get_dataset_details(dataset_id)

        if "error" in result:
            typer.echo(f"Error [{dataset_id}]: {result['error']}")
            continue

        typer.echo(f"\n{dataset_id}")
        typer.echo("=" * len(dataset_id))

        # Show install status
        if dataset_id in local_datasets:
            local_info = local_datasets[dataset_id]
            if local_info.get('synced'):
                typer.echo("Status: Installed")
            else:
                typer.echo("Status: Added (not synced)")
        else:
            typer.echo("Status: Not installed")

        if result.get('title'):
            typer.echo(f"\nTitle: {result['title']}")
        if result.get('description'):
            desc = result['description'][:200]
            typer.echo(f"\nDescription:\n  {desc}...")
        if result.get('row_count'):
            typer.echo(f"\nRows: {result['row_count']:,}")
        if result.get('size_bytes'):
            typer.echo(f"Size: {format_bytes(result['size_bytes'])}")

        schema = result.get('schema', [])
        if schema:
            typer.echo(f"\nSchema ({len(schema)} columns):")
            for col in schema[:20]:
                col_type = col.get('type', 'unknown')
                col_name = col.get('name', 'unknown')
                typer.echo(f"  {col_name}: {col_type}")
            if len(schema) > 20:
                typer.echo(f"  ... and {len(schema) - 20} more columns")

        if result.get('source_name'):
            typer.echo(f"\nSource: {result['source_name']}")
        if result.get('license'):
            typer.echo(f"License: {result['license']}")

        if dataset_id not in local_datasets:
            typer.echo(f"\nUse 'subsets add {dataset_id}' to download this dataset.")


# =============================================================================
# LOCAL COLLECTION COMMANDS
# =============================================================================

@app.command(name="list")
def list_datasets(
    query: Optional[str] = typer.Argument(None, help="Optional keyword to filter datasets")
):
    """List installed datasets in your local collection"""
    datasets = get_local_platform().list_local_datasets()
    synced = [d for d in datasets if d.get('synced', False)]

    # Filter by query if provided
    if query:
        query_lower = query.lower()
        synced = [d for d in synced if query_lower in d['id'].lower()]

    if not synced:
        if query:
            typer.echo(f"No installed datasets matching '{query}'")
        else:
            typer.echo("No datasets installed.")
            typer.echo("Use 'subsets search <query>' to find datasets.")
            typer.echo("Use 'subsets add <dataset_id>' to install one.")
        return

    header = f"Installed datasets matching '{query}'" if query else "Installed datasets"
    typer.echo(f"{header} ({len(synced)}):\n")
    for d in synced:
        size = format_bytes(d.get('size_bytes', 0) or 0)
        rows = d.get('row_count', 0)
        row_str = f"{rows:,} rows" if rows else ""
        typer.echo(f"  {d['id']}")
        if row_str or size:
            typer.echo(f"    {row_str}, {size}")


@app.command()
def add(
    dataset_ids: List[str] = typer.Argument(None, help="Dataset IDs to add"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="File with dataset IDs (one per line)")
):
    """Download datasets to your local collection"""
    from sync import sync_all_datasets
    from tqdm import tqdm
    import requests
    from auth import get_api_key, get_api_url

    # Collect dataset IDs from arguments and/or file
    ids_to_add = list(dataset_ids) if dataset_ids else []

    if file:
        if not file.exists():
            typer.echo(f"Error: File not found: {file}")
            raise typer.Exit(1)
        file_ids = [line.strip() for line in file.read_text().strip().split("\n") if line.strip()]
        ids_to_add.extend(file_ids)

    if not ids_to_add:
        typer.echo("No dataset IDs provided.")
        typer.echo("Usage: subsets add <dataset_id>")
        typer.echo("       subsets add --file datasets.txt")
        raise typer.Exit(1)

    # Remove duplicates while preserving order
    ids_to_add = list(dict.fromkeys(ids_to_add))

    api_key = get_api_key()
    api_url = get_api_url()

    if not api_key:
        typer.echo("Error: No API key configured")
        typer.echo("Run 'subsets login' to set up your API key")
        raise typer.Exit(1)

    # Batch fetch metadata
    typer.echo(f"Fetching metadata for {len(ids_to_add)} dataset(s)...")
    all_metadata = {}
    try:
        response = requests.post(
            f"{api_url}/datasets/batch-metadata",
            json={"dataset_ids": ids_to_add},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30
        )

        if response.ok:
            data = response.json()
            for dataset_id, metadata in data.get("datasets", {}).items():
                all_metadata[dataset_id] = metadata
        else:
            typer.echo(f"Error: API request failed with status {response.status_code}")
            raise typer.Exit(1)
    except requests.exceptions.RequestException as e:
        typer.echo(f"Error: Could not connect to API: {e}")
        raise typer.Exit(1)

    # Build dataset entries
    datasets = []
    total_size_bytes = 0
    not_found = []

    for dataset_id in ids_to_add:
        metadata = all_metadata.get(dataset_id)
        if metadata:
            size_bytes = metadata.get('size_bytes') or 0
            total_size_bytes += size_bytes
            datasets.append({
                'id': dataset_id,
                'name': dataset_id,
                'added_at': datetime.now().isoformat(),
                'size_bytes': size_bytes,
                'row_count': metadata.get('row_count'),
                'delta_version': metadata.get('delta_version'),
                'last_modified': metadata.get('last_modified')
            })
        else:
            not_found.append(dataset_id)

    if not_found:
        typer.echo(f"Not found: {', '.join(not_found)}")

    if not datasets:
        typer.echo("No valid datasets to add")
        raise typer.Exit(1)

    # Add to collection
    get_local_platform().add_datasets(datasets)

    typer.echo(f"Downloading {len(datasets)} dataset(s) ({format_bytes(total_size_bytes)})...")

    # Sync with progress bar
    data_path = get_local_platform().data_path

    pbar = tqdm(
        total=total_size_bytes,
        desc="Downloading",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=True
    )

    def progress_callback(completed_datasets, total_datasets, completed_bytes, total_bytes):
        pbar.n = completed_bytes
        pbar.set_postfix_str(f"{completed_datasets}/{total_datasets} datasets")
        pbar.refresh()

    results = sync_all_datasets(
        datasets,
        data_path,
        progress_callback=progress_callback,
        max_workers=8
    )

    pbar.close()

    # Summary
    synced = len(results['synced'])
    up_to_date = len(results['up_to_date'])
    failed = len(results['failed'])

    typer.echo(f"\nInstalled: {synced}")
    if up_to_date:
        typer.echo(f"Already installed: {up_to_date}")
    if failed:
        typer.echo(f"Failed: {failed}")
        for fail in results['failed']:
            typer.echo(f"  - {fail['dataset_id']}: {fail.get('error', 'Unknown')}")

    typer.echo("\nDatasets ready to query with 'subsets query'")


@app.command()
def remove(
    dataset_ids: List[str] = typer.Argument(..., help="Dataset IDs to remove")
):
    """Remove datasets from your local collection"""
    if len(dataset_ids) == 1:
        confirm_msg = f"Remove dataset '{dataset_ids[0]}' and delete its data?"
    else:
        confirm_msg = f"Remove {len(dataset_ids)} datasets and delete their data?"

    if not typer.confirm(confirm_msg):
        typer.echo("Cancelled")
        return

    local = get_local_platform()
    removed_count = 0
    bytes_freed = 0

    for dataset_id in dataset_ids:
        dataset_path = local.data_path / dataset_id

        # Calculate size before deletion
        if dataset_path.exists():
            for item in dataset_path.rglob('*'):
                if item.is_file():
                    bytes_freed += item.stat().st_size
            shutil.rmtree(dataset_path)

        # Remove from collection
        count = local.remove_datasets([dataset_id])
        removed_count += count

    typer.echo(f"Removed {removed_count} dataset(s), freed {format_bytes(bytes_freed)}")


@app.command()
def sync(
    dataset_ids: Optional[List[str]] = typer.Argument(None, help="Dataset IDs to sync (default: all)")
):
    """Update installed datasets to latest versions"""
    from sync import sync_all_datasets
    from tqdm import tqdm

    data_path = get_local_platform().data_path
    all_datasets = get_local_platform().list_local_datasets()

    if not all_datasets:
        typer.echo("No datasets installed. Use 'subsets add' to install datasets.")
        return

    # Filter if specific IDs provided
    if dataset_ids:
        datasets_to_sync = [d for d in all_datasets if d['id'] in dataset_ids]
        not_found = set(dataset_ids) - {d['id'] for d in datasets_to_sync}
        if not_found:
            typer.echo(f"Not installed: {', '.join(not_found)}")
        if not datasets_to_sync:
            return
    else:
        datasets_to_sync = all_datasets

    # Calculate total size
    total_size = sum(d.get('size_bytes', 0) or 0 for d in datasets_to_sync)

    typer.echo(f"Syncing {len(datasets_to_sync)} dataset(s)...")

    pbar = tqdm(
        total=total_size,
        desc="Syncing",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=True
    )

    def progress_callback(completed_datasets, total_datasets, completed_bytes, total_bytes):
        pbar.n = completed_bytes
        pbar.set_postfix_str(f"{completed_datasets}/{total_datasets} datasets")
        pbar.refresh()

    results = sync_all_datasets(
        datasets_to_sync,
        data_path,
        progress_callback=progress_callback,
        max_workers=8
    )

    pbar.close()

    typer.echo(f"\nUpdated: {len(results['synced'])}")
    if results['up_to_date']:
        typer.echo(f"Already up to date: {len(results['up_to_date'])}")
    if results['failed']:
        typer.echo(f"Failed: {len(results['failed'])}")
        for fail in results['failed']:
            typer.echo(f"  - {fail['dataset_id']}: {fail.get('error', 'Unknown')}")


# =============================================================================
# QUERY COMMAND (always local)
# =============================================================================

@app.command()
def query(
    sql: str = typer.Argument(..., help="SQL query to execute"),
    format: str = typer.Option("tsv", "--format", "-f", help="Output format: tsv or json")
):
    """Execute a SQL query against your installed datasets"""
    result = get_local_platform().execute_sql_query(sql, output_format=format)

    if "error" in result:
        typer.echo(f"Error: {result['error']}")
        if "suggestion" in result:
            typer.echo(f"Hint: {result['suggestion']}")
        raise typer.Exit(1)

    # Output results
    if format == "json":
        typer.echo(json.dumps(result, indent=2))
    else:
        # TSV format
        if "content" in result:
            typer.echo(result["content"])
        elif "columns" in result and "rows" in result:
            columns = result["columns"]
            rows = result["rows"]
            typer.echo("\t".join(str(c) for c in columns))
            for row in rows:
                typer.echo("\t".join(str(v) if v is not None else "" for v in row))

    # Show row count
    row_count = result.get("row_count", 0)
    if row_count:
        typer.echo(f"\n({row_count} rows)", err=True)


# =============================================================================
# STATUS COMMAND
# =============================================================================

@app.command()
def status():
    """Show current status"""
    from auth import get_api_key

    api_key = get_api_key()
    local = get_local_platform()

    typer.echo(f"API key: {'configured' if api_key else 'not set'}")
    typer.echo(f"Data directory: {local.data_path}")

    # Count local datasets
    datasets = local.list_local_datasets()
    synced_count = sum(1 for d in datasets if d.get('synced', False))

    typer.echo(f"\nInstalled datasets: {synced_count}")

    # Calculate disk usage
    total_bytes = 0
    if local.data_path.exists():
        for item in local.data_path.rglob('*'):
            if item.is_file():
                total_bytes += item.stat().st_size
    typer.echo(f"Disk usage: {format_bytes(total_bytes)}")


def main():
    """Main entry point for the CLI"""
    app()


if __name__ == "__main__":
    main()
