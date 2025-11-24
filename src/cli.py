#!/usr/bin/env python3
"""Subsets CLI using Typer - replacement for Click-based CLI to fix batch fetching issues"""
import typer
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import sqlite3
import shutil
from enum import Enum

# Initialize Typer app
app = typer.Typer(help="Subsets CLI for managing local and remote data platform")


# Platform initialization moved to functions to avoid module-level side effects
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


class Mode(str, Enum):
    local = "local"
    remote = "remote"


@app.command()
def init():
    """Initialize Subsets with your API key"""
    from auth import save_api_key, get_api_key
    
    # Check if API key already exists
    existing_key = get_api_key()
    if existing_key:
        typer.echo(f"Current API key: {existing_key[:7]}...")
        if not typer.confirm("Do you want to update your API key?"):
            return
    
    # Prompt for API key
    api_key = typer.prompt("Enter your Subsets API key", hide_input=True)
    
    # Basic validation
    if not api_key.startswith("sk_"):
        typer.echo("Warning: API keys usually start with 'sk_'")
        if not typer.confirm("Continue anyway?"):
            return
    
    # Save the API key
    save_api_key(api_key)
    typer.echo("✓ API key saved successfully!")
    typer.echo("You can now add and sync datasets from the catalog.")


@app.command()
def use(mode: Mode):
    """Switch between local and remote mode"""
    from auth import get_api_key
    
    config_path = Path.home() / "subsets" / "config.json"
    config = {"mode": mode.value}
    
    if mode == Mode.remote and not get_api_key():
        typer.echo("Warning: Remote mode requires an API key")
        typer.echo("Run 'subsets init' to configure your API key")
        return
    
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    typer.echo(f"Switched to {mode.value} mode")


@app.command()
def add(dataset_ids: List[str] = typer.Argument(..., help="Dataset IDs to add")):
    """Add one or more datasets to the local collection"""
    # Convert dataset IDs to dataset objects
    datasets = []
    total_size_bytes = 0
    
    # Batch fetch metadata
    if dataset_ids:
        # Now we can safely import and use batch fetching with Typer!
        import requests
        from auth import get_api_key, get_api_url

        api_key = get_api_key()
        api_url = get_api_url()

        if not api_key:
            typer.echo("Error: No API key configured")
            typer.echo("Run 'subsets init' to set up your API key")
            raise typer.Exit(1)

        all_metadata = {}
        try:
            response = requests.post(
                f"{api_url}/datasets/batch-metadata",
                json={"dataset_ids": list(dataset_ids)},
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10
            )

            if response.ok:
                data = response.json()
                for dataset_id, metadata in data.get("datasets", {}).items():
                    all_metadata[dataset_id] = metadata
            else:
                typer.echo(f"Error: API request failed with status {response.status_code}")
                typer.echo(f"Response: {response.text}")
                raise typer.Exit(1)
        except requests.exceptions.RequestException as e:
            typer.echo(f"Error: Could not connect to API: {e}")
            raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"Error: Could not fetch metadata: {e}")
            raise typer.Exit(1)
        
        valid_count = 0
        not_found_count = 0
        empty_count = 0
        
        for dataset_id in dataset_ids:
            dataset_info = {
                'id': dataset_id,
                'name': dataset_id,
                'added_at': datetime.now().isoformat()
            }
            
            metadata = all_metadata.get(dataset_id)
            if metadata:
                # Keep NULL as None to distinguish from 0
                size_bytes = metadata.get('size_bytes')
                row_count = metadata.get('row_count')

                dataset_info.update({
                    'size_bytes': size_bytes,
                    'row_count': row_count,
                    'delta_version': metadata.get('delta_version'),
                    'last_modified': metadata.get('last_modified')
                })

                # Add to total if size is known
                if size_bytes is not None:
                    total_size_bytes += size_bytes

                # Dataset is valid if it has metadata
                valid_count += 1

                datasets.append(dataset_info)
            else:
                not_found_count += 1

        # Show summary only if there are issues
        if not_found_count > 0:
            typer.echo(f"✗ {not_found_count} datasets not found")
    
    # Add datasets to local collection
    stats = get_local_platform().add_datasets(datasets)
    
    # Only print if datasets were actually added
    if stats['new'] > 0:
        size_gb = total_size_bytes / (1024 ** 3)
        if size_gb >= 1:
            typer.echo(f"Added {stats['new']} datasets ({size_gb:.2f} GB)")
        else:
            size_mb = total_size_bytes / (1024 ** 2)
            if size_mb >= 1:
                typer.echo(f"Added {stats['new']} datasets ({size_mb:.0f} MB)")
            else:
                size_kb = total_size_bytes / 1024
                typer.echo(f"Added {stats['new']} datasets ({size_kb:.0f} KB)")
        typer.echo("Run 'subsets sync' to download the data")


@app.command(name="list")
def list_datasets():
    """List all datasets in your local collection"""
    datasets = get_local_platform().list_local_datasets()
    
    if not datasets:
        typer.echo("No datasets found in your collection")
        return
    
    typer.echo(f"Datasets in your collection ({len(datasets)} total):")
    for dataset in datasets:
        synced_status = "✓" if dataset.get('synced', False) else "○"
        typer.echo(f"  {synced_status} {dataset['id']}")
        if dataset.get('description'):
            typer.echo(f"    {dataset['description']}")


@app.command()
def remove(
    dataset_ids: List[str] = typer.Argument(..., help="Dataset IDs to remove")
):
    """Remove datasets from the local collection
    
    Examples:
        subsets remove dataset1
        subsets remove dataset1 dataset2 dataset3
    """
    if len(dataset_ids) == 1:
        confirm_msg = f"Are you sure you want to remove dataset '{dataset_ids[0]}'?"
    else:
        confirm_msg = f"Are you sure you want to remove {len(dataset_ids)} datasets?"
    
    if typer.confirm(confirm_msg):
        removed_count = get_local_platform().remove_datasets(list(dataset_ids))
        if removed_count == 1:
            typer.echo(f"Removed 1 dataset")
        else:
            typer.echo(f"Removed {removed_count} datasets")
    else:
        typer.echo("Cancelled")


@app.command()
def sync():
    """Sync all datasets from the remote catalog"""
    from sync import sync_all_datasets
    from tqdm import tqdm
    
    data_path = get_local_platform().data_path
    
    # Get all datasets in collection
    datasets = get_local_platform().list_local_datasets()
    
    if not datasets:
        typer.echo("No datasets in your collection. Use 'subsets add' to add datasets.")
        return
    
    # Find unsynced datasets
    unsynced = [d for d in datasets if not d.get('synced', False)]
    
    if not unsynced:
        typer.echo(f"✓ All {len(datasets)} datasets are already synced")
        return
    
    # Calculate total size
    total_size = sum(d.get('size_bytes', 0) or 0 for d in unsynced)
    
    # Format size for display
    if total_size >= 1024**3:
        size_str = f"{total_size / (1024**3):.2f} GB"
    elif total_size >= 1024**2:
        size_str = f"{total_size / (1024**2):.0f} MB"
    else:
        size_str = f"{total_size / 1024:.0f} KB"
    
    # Start syncing message
    typer.echo(f"Syncing {len(unsynced)} datasets ({size_str})...")
    
    # Create single progress bar showing data transfer with dataset count
    pbar = tqdm(
        total=total_size,
        desc="Syncing",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=True
    )
    
    def progress_callback(completed_datasets, total_datasets, completed_bytes, total_bytes, indexed_count):
        # Update progress bar
        pbar.n = completed_bytes
        # Update description to show dataset count
        pbar.set_postfix_str(f"{completed_datasets}/{total_datasets} datasets")
        pbar.refresh()

    # Get embedding service from local platform
    local_platform = get_local_platform()

    # Start syncing with parallel workers
    results = sync_all_datasets(
        unsynced,
        data_path,
        embedding_service=local_platform.embedding_service,
        progress_callback=progress_callback,
        max_workers=8  # Use 8 parallel workers
    )
    
    # Close progress bar
    pbar.close()

    # Simple summary with indexing info
    indexed_count = results.get('indexed_count', 0)
    synced_count = len(results['synced'])

    if indexed_count > 0:
        typer.echo(f"\n✓ Synced and indexed {synced_count} datasets")
    else:
        typer.echo(f"\n✓ Synced: {synced_count}")

    if results['up_to_date']:
        typer.echo(f"✓ Already up to date: {len(results['up_to_date'])}")
    if results['failed']:
        typer.echo(f"✗ Failed: {len(results['failed'])}")
        for fail in results['failed']:
            typer.echo(f"  - {fail['dataset_id']}: {fail.get('error', 'Unknown error')}")


@app.command()
def status():
    """Show current status and configuration"""
    from sync import get_local_catalog
    
    # Check current mode
    config_path = Path.home() / "subsets" / "config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            mode = config.get("mode", "local")
    else:
        mode = "local"
    
    typer.echo(f"Current mode: {mode}")
    typer.echo(f"Base directory: {get_local_platform().base_path}")
    
    # Count local datasets
    datasets = get_local_platform().list_local_datasets()
    synced_count = sum(1 for d in datasets if d.get('synced', False))
    
    typer.echo(f"Total datasets: {len(datasets)}")
    typer.echo(f"Synced datasets: {synced_count}")
    typer.echo(f"Pending sync: {len(datasets) - synced_count}")
    
    # Check local Iceberg catalog
    try:
        catalog = get_local_catalog()
        tables = [table.name for table in catalog.list_tables("subsets")]
        typer.echo(f"Iceberg catalog tables: {len(tables)}")
    except Exception:
        typer.echo(f"Iceberg catalog: Not initialized")
    
    # Show recent queries
    conn = sqlite3.connect(get_local_platform().log_path)
    cursor = conn.execute("""
        SELECT COUNT(*) as count, mode
        FROM query_log
        WHERE timestamp > datetime('now', '-7 days')
        GROUP BY mode
    """)
    
    typer.echo("\nRecent queries (last 7 days):")
    for row in cursor.fetchall():
        typer.echo(f"  {row[1]} mode: {row[0]} queries")
    conn.close()


@app.command()
def logs():
    """Show recent query logs"""
    conn = sqlite3.connect(get_local_platform().log_path)
    
    # Just show last 10 queries
    query = """
        SELECT timestamp, mode, query, success, error_message, rows_returned
        FROM query_log
        ORDER BY timestamp DESC LIMIT 10
    """

    cursor = conn.execute(query)
    rows = cursor.fetchall()

    if not rows:
        typer.echo("No queries found")
    else:
        for row in rows:
            timestamp, mode, query_text, success, error, rows_ret = row
            status = "✓" if success else "✗"
            typer.echo(f"\n[{timestamp}] {status} {mode}")
            typer.echo(f"  Query: {query_text[:100]}...")
            if rows_ret:
                typer.echo(f"  Rows: {rows_ret}")
            if error:
                typer.echo(f"  Error: {error}")
    
    conn.close()


@app.command()
def cleanup():
    """Remove local data files for datasets that are no longer in your collection"""
    # Get all datasets in collection
    collection_datasets = get_local_platform().list_local_datasets()
    collection_dataset_ids = {d['id'] for d in collection_datasets}
    
    # Get all local data directories
    local_data_dirs = set()
    for item in get_local_platform().data_path.glob("*"):
        if item.is_dir() and not item.name.startswith('.'):
            local_data_dirs.add(item.name)
    
    # Find orphaned data directories
    to_remove = local_data_dirs - collection_dataset_ids
    
    if not to_remove:
        typer.echo("✓ No orphaned data to clean up")
        return
    
    typer.echo(f"Found {len(to_remove)} orphaned dataset(s). Cleaning up...")
    
    # Just remove them
    removed_count = 0
    for dataset_id in to_remove:
        dataset_path = get_local_platform().data_path / dataset_id
        try:
            shutil.rmtree(dataset_path)
            removed_count += 1
            typer.echo(f"  ✓ Removed {dataset_id}")
        except Exception as e:
            typer.echo(f"  ✗ Failed to remove {dataset_id}: {e}")
    
    typer.echo(f"\n✓ Cleaned up {removed_count} dataset(s)")


def main():
    """Main entry point for the CLI"""
    app()


if __name__ == "__main__":
    main()