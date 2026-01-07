import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def get_dataset_metadata(dataset_id: str) -> Optional[Dict[str, Any]]:
    """Fetch metadata for a single dataset from the server."""
    result = get_batch_metadata([dataset_id])
    return result.get(dataset_id) if result else None


def get_batch_metadata(dataset_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch metadata for multiple datasets from the server."""
    from auth import get_api_key, get_api_url

    api_key = get_api_key()
    api_url = get_api_url()

    if not api_key or not dataset_ids:
        return {}

    try:
        response = requests.post(
            f"{api_url}/datasets/batch-metadata",
            json={"dataset_ids": dataset_ids},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30
        )

        if response.ok:
            data = response.json()
            result = {}
            for dataset_id, metadata in data.get("datasets", {}).items():
                result[dataset_id] = {
                    'dataset_id': dataset_id,
                    'row_count': metadata.get('row_count', 0),
                    'size_bytes': metadata.get('size_bytes', 0) or 0,
                    'delta_version': metadata.get('delta_version'),
                    'last_modified': metadata.get('last_modified')
                }
            return result
        else:
            print(f"Failed to fetch batch metadata: {response.status_code} - {response.text}")
            return {}

    except Exception as e:
        print(f"Error fetching batch metadata: {e}")
        return {}


def sync_dataset(
    dataset_id: str,
    data_path: Path,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    quiet: bool = False
) -> Dict[str, Any]:
    """
    Sync a dataset using Delta Lake incremental sync.
    Downloads only changed files using the versioned download endpoint.
    """
    from auth import get_api_key, get_api_url

    dataset_path = data_path / dataset_id
    dataset_path.mkdir(exist_ok=True, parents=True)

    stats = {
        "dataset_id": dataset_id,
        "status": "started",
        "files_downloaded": 0,
        "bytes_downloaded": 0,
        "error": None
    }

    try:
        api_key = get_api_key()
        api_url = get_api_url()

        if not api_key:
            stats["status"] = "failed"
            stats["error"] = "No API key configured. Run 'subsets init' to configure."
            return stats

        # Load local sync info
        sync_info_path = dataset_path / "sync_info.json"
        local_version = None

        if sync_info_path.exists():
            with open(sync_info_path, 'r') as f:
                sync_info = json.load(f)
                local_version = sync_info.get("delta_version")

        # Get remote version
        metadata = get_dataset_metadata(dataset_id)
        if not metadata:
            stats["status"] = "failed"
            stats["error"] = f"Dataset not found: {dataset_id}"
            return stats

        remote_version = metadata.get('delta_version')

        # Check if update needed
        if local_version == remote_version:
            stats["status"] = "up_to_date"
            stats["delta_version"] = remote_version
            return stats

        if not quiet:
            if local_version is None:
                print(f"Initial sync: {dataset_id} (version {remote_version})")
            else:
                print(f"Incremental sync: {dataset_id} (v{local_version} → v{remote_version})")

        # Request download URLs
        payload = {}
        if local_version is not None:
            payload["from_version"] = local_version
            payload["to_version"] = remote_version

        response = requests.post(
            f"{api_url}/datasets/{dataset_id}/download-urls",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60
        )

        if not response.ok:
            stats["status"] = "failed"
            stats["error"] = f"Failed to get download URLs: {response.status_code} - {response.text}"
            return stats

        urls_data = response.json()
        files = urls_data['files']

        if not quiet:
            print(f"Downloading {len(files)} files ({urls_data['total_bytes']:,} bytes)...")

        # Download each file
        for file_info in files:
            file_path = dataset_path / file_info['path']
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file from presigned URL
            file_response = requests.get(file_info['url'], timeout=300)
            file_response.raise_for_status()

            with open(file_path, 'wb') as f:
                f.write(file_response.content)

            stats["files_downloaded"] += 1
            stats["bytes_downloaded"] += file_info['size_bytes']

        # Apply incremental update if Delta table exists
        if local_version is not None:
            try:
                from deltalake import DeltaTable
                dt = DeltaTable(str(dataset_path))
                dt.update_incremental()

                if not quiet:
                    print(f"Applied incremental Delta update to v{remote_version}")
            except Exception as e:
                if not quiet:
                    print(f"Warning: Failed to apply Delta update: {e}")

        # Save sync info
        sync_metadata = {
            "dataset_id": dataset_id,
            "delta_version": remote_version,
            "last_modified": metadata.get('last_modified'),
            "sync_timestamp": datetime.now().isoformat(),
            "files_count": len(files),
            "bytes_synced": stats["bytes_downloaded"]
        }

        with open(sync_info_path, 'w') as f:
            json.dump(sync_metadata, f, indent=2)

        stats["status"] = "completed"
        stats["delta_version"] = remote_version

        if not quiet:
            print(f"✓ Synced {dataset_id} to v{remote_version}")

    except Exception as e:
        stats["status"] = "failed"
        stats["error"] = str(e)
        if not quiet:
            print(f"✗ Sync failed for {dataset_id}: {e}")

    return stats


def sync_all_datasets(
    datasets: List[Dict[str, Any]],
    data_path: Path,
    progress_callback=None,
    max_workers: int = 8
) -> Dict[str, List[Dict[str, Any]]]:
    """Sync all datasets in parallel with progress tracking."""
    results = {
        "synced": [],
        "failed": [],
        "up_to_date": []
    }
    results_lock = threading.Lock()

    total_datasets = len(datasets)
    total_size_bytes = sum(d.get('size_bytes', 0) or 0 for d in datasets)

    # Progress tracking
    completed_count = 0
    completed_bytes = 0
    progress_lock = threading.Lock()

    def sync_with_progress(dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper to handle progress updates."""
        nonlocal completed_count, completed_bytes

        dataset_id = dataset['id']
        dataset_size = dataset.get('size_bytes', 0) or 0

        # Perform the sync quietly
        sync_result = sync_dataset(
            dataset_id,
            data_path,
            dataset_metadata=dataset,
            quiet=True
        )

        # Update progress
        with progress_lock:
            completed_count += 1
            if sync_result["status"] in ["completed", "up_to_date"]:
                completed_bytes += sync_result.get("bytes_downloaded", 0)

            if progress_callback:
                progress_callback(
                    completed_count,
                    total_datasets,
                    completed_bytes,
                    total_size_bytes
                )

        # Store result
        with results_lock:
            if sync_result["status"] == "completed":
                results["synced"].append(sync_result)
            elif sync_result["status"] == "up_to_date":
                results["up_to_date"].append(sync_result)
            else:
                results["failed"].append(sync_result)

        return sync_result

    # Execute in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_dataset = {
            executor.submit(sync_with_progress, dataset): dataset
            for dataset in datasets
        }

        for future in as_completed(future_to_dataset):
            dataset = future_to_dataset[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error syncing {dataset['id']}: {e}")
                with results_lock:
                    results["failed"].append({
                        "dataset_id": dataset['id'],
                        "status": "failed",
                        "error": str(e)
                    })

    return results
