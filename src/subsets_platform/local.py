import os
import json
import sqlite3
import duckdb
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import hashlib


class LocalDataPlatform:
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or (Path.home() / "subsets")
        self.data_path = self.base_path / "data"
        self.log_path = self.base_path / "query_log.db"
        self.config_path = self.base_path / "config.json"

        self._ensure_directories()
        self._init_query_log()
        self._load_config()

        # Initialize keyword search (downloads index from R2 on first use)
        from keyword_search import get_keyword_search
        self.keyword_search = get_keyword_search(self.base_path)
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        self.base_path.mkdir(exist_ok=True)
        self.data_path.mkdir(exist_ok=True)
    
    def _init_query_log(self):
        """Initialize the query log database"""
        conn = sqlite3.connect(self.log_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT NOT NULL,
                mode TEXT NOT NULL,
                dataset TEXT,
                success BOOLEAN,
                error_message TEXT,
                rows_returned INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def _load_config(self):
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "mode": "local",
                "datasets": []
            }
            self._save_config()
    
    def _save_config(self):
        """Save configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _log_query(self, query: str, dataset: Optional[str] = None,
                   success: bool = True, error_message: Optional[str] = None,
                   rows_returned: Optional[int] = None):
        """Log a query execution"""
        conn = sqlite3.connect(self.log_path)
        conn.execute("""
            INSERT INTO query_log (query, mode, dataset, success, error_message, rows_returned)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (query, "local", dataset, success, error_message, rows_returned))
        conn.commit()
        conn.close()
    
    def _get_dataset_hash(self, dataset_name: str) -> str:
        """Get a safe filename hash for a dataset"""
        return hashlib.md5(dataset_name.encode()).hexdigest()
    
    def _get_all_datasets(self) -> List[Dict[str, Any]]:
        """Get all datasets from the local collection"""
        return self.config.get('datasets', [])
    
    def list_datasets(
        self,
        limit: int = 10,
        offset: int = 0,
        license: Optional[str] = None,
        git_user: Optional[str] = None,
        q: Optional[str] = None,
        min_score: Optional[float] = None,
        detailed: bool = False
    ) -> Dict[str, Any]:
        """List available datasets from the local collection using keyword search."""
        try:
            all_datasets = self._get_all_datasets()

            # If query provided, use keyword search for ranking
            if q and self.keyword_search:
                try:
                    # Get ranked results from keyword search
                    search_results = self.keyword_search.search(q, limit=100)
                    ranked_ids = {r["dataset_id"]: r["score"] for r in search_results}

                    # Filter and rank datasets
                    filtered_datasets = []
                    for dataset in all_datasets:
                        dataset_id = dataset.get("id")
                        if dataset_id not in ranked_ids:
                            continue
                        if license and dataset.get("license") != license:
                            continue
                        if git_user and dataset.get("git_user") != git_user:
                            continue
                        if min_score and ranked_ids[dataset_id] < min_score:
                            continue

                        # Add score to dataset for sorting
                        dataset_with_score = {**dataset, "_score": ranked_ids[dataset_id]}
                        filtered_datasets.append(dataset_with_score)

                    # Sort by score descending
                    filtered_datasets.sort(key=lambda d: d.get("_score", 0), reverse=True)

                    # Remove internal score field
                    for d in filtered_datasets:
                        d.pop("_score", None)

                except Exception as e:
                    print(f"Keyword search failed: {e}")
                    filtered_datasets = []
            else:
                # No query - just filter
                filtered_datasets = []
                for dataset in all_datasets:
                    if license and dataset.get("license") != license:
                        continue
                    if git_user and dataset.get("git_user") != git_user:
                        continue
                    filtered_datasets.append(dataset)

            # Apply pagination
            total = len(filtered_datasets)
            paginated = filtered_datasets[offset:offset + limit]

            self._log_query(
                f"list_datasets(q={q}, limit={limit}, offset={offset})",
                success=True,
                rows_returned=len(paginated)
            )

            return {
                "datasets": paginated,
                "total": total,
                "limit": limit,
                "offset": offset
            }

        except Exception as e:
            self._log_query(
                f"list_datasets(q={q}, limit={limit}, offset={offset})",
                success=False,
                error_message=str(e)
            )
            return {
                "error": f"Failed to list datasets: {str(e)}"
            }
    
    def execute_sql_query(self, query: str, output_format: str = "json") -> Dict[str, Any]:
        """Execute a SQL query against local datasets using DuckDB"""
        try:
            start_time = datetime.now()
            
            # Initialize DuckDB connection
            conn = duckdb.connect(':memory:')
            
            registered_tables = []

            # Register datasets from data directory
            all_datasets = self._get_all_datasets()

            for dataset in all_datasets:
                dataset_id = dataset.get("id")
                if not dataset_id:
                    continue

                dataset_dir = self.data_path / dataset_id

                # Skip if not synced
                if not dataset_dir.exists():
                    continue

                table_name = dataset_id.replace("-", "_").replace(".", "_")

                try:
                    # Try Delta Lake table first (preferred)
                    delta_log_dir = dataset_dir / "_delta_log"
                    if delta_log_dir.exists():
                        conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM delta_scan('{dataset_dir}')")
                        registered_tables.append(table_name)
                        continue

                    # Fallback: single parquet file
                    parquet_file = dataset_dir / "data.parquet"
                    if parquet_file.exists():
                        conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{parquet_file}')")
                        registered_tables.append(table_name)
                        continue

                    # Fallback: hashed filename (legacy)
                    dataset_hash = self._get_dataset_hash(dataset_id)
                    for ext in ['.parquet', '.csv', '.json']:
                        potential_file = self.data_path / f"{dataset_hash}{ext}"
                        if potential_file.exists():
                            if ext == '.parquet':
                                conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{potential_file}')")
                            elif ext == '.csv':
                                conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_csv_auto('{potential_file}')")
                            elif ext == '.json':
                                conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_json_auto('{potential_file}')")
                            registered_tables.append(table_name)
                            break

                except Exception as e:
                    print(f"Failed to register {dataset_id}: {e}")
                    continue
            
            # Execute the query
            result = conn.execute(query)
            
            # Get column names and data directly without pandas
            columns = [desc[0] for desc in result.description]
            data = result.fetchall()
            
            # Format output to match remote format
            if output_format == "tsv":
                # Create TSV format
                tsv_lines = []
                # Header row
                tsv_lines.append('\t'.join(columns))
                # Data rows
                for row in data:
                    # Convert each value to string, handling None as empty string
                    row_str = '\t'.join(str(val) if val is not None else '' for val in row)
                    tsv_lines.append(row_str)

                tsv_content = '\n'.join(tsv_lines)

                result_dict = {
                    "format": "tsv",
                    "content": tsv_content,
                    "row_count": len(data),
                    "column_count": len(columns),
                    "render_hint": "markdown_table",
                    "instructions": "This TSV data should be rendered as a markdown table"
                }
            else:
                # JSON format - match remote structure (columns + rows)
                data_as_lists = [list(row) for row in data]

                result_dict = {
                    "columns": columns,
                    "rows": data_as_lists,
                    "row_count": len(data_as_lists)
                }

            self._log_query(
                query,
                success=True,
                rows_returned=len(data)
            )

            conn.close()

            # Return result directly (like remote does)
            return result_dict
            
        except Exception as e:
            self._log_query(
                query,
                success=False,
                error_message=str(e)
            )
            return {
                "error": f"Failed to execute SQL query: {str(e)}"
            }
    
    def add_datasets(self, datasets: List[Dict[str, Any]]) -> Dict[str, int]:
        """Add datasets to the local collection"""
        if 'datasets' not in self.config:
            self.config['datasets'] = []
        
        existing_ids = {d['id'] for d in self.config['datasets']}
        new_count = 0
        existing_count = 0
        
        for dataset in datasets:
            if dataset['id'] in existing_ids:
                existing_count += 1
            else:
                self.config['datasets'].append(dataset)
                new_count += 1
        
        if new_count > 0:
            self._save_config()
        
        return {
            'new': new_count,
            'existing': existing_count
        }
    
    def list_local_datasets(self) -> List[Dict[str, Any]]:
        """List all datasets in the local collection"""
        datasets = self.config.get('datasets', [])

        # Check sync status and version by reading sync_info.json
        for dataset in datasets:
            dataset_path = self.data_path / dataset['id']
            sync_info_path = dataset_path / 'sync_info.json'

            if sync_info_path.exists():
                try:
                    with open(sync_info_path, 'r') as f:
                        sync_info = json.load(f)
                    dataset['synced'] = True
                    dataset['local_version'] = sync_info.get('delta_version')
                    dataset['last_synced'] = sync_info.get('sync_timestamp')
                except Exception:
                    dataset['synced'] = dataset_path.exists() and any(dataset_path.iterdir())
            else:
                dataset['synced'] = False

        return datasets
    
    def remove_datasets(self, dataset_ids: List[str]) -> int:
        """Remove datasets from the local collection"""
        if 'datasets' not in self.config:
            return 0

        original_count = len(self.config['datasets'])
        self.config['datasets'] = [
            d for d in self.config['datasets']
            if d['id'] not in dataset_ids
        ]

        removed_count = original_count - len(self.config['datasets'])

        if removed_count > 0:
            self._save_config()

        return removed_count

    def search_datasets(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """
        Fast keyword search returning dataset IDs with available metadata.

        Uses the same pre-built index as the remote API for consistent results.
        Downloads the index from R2 on first use.

        Local mode returns less metadata than remote (no description/schema in local index).
        """
        if self.keyword_search is None:
            return {
                "error": "Keyword search not available",
                "query": query,
                "results": [],
                "total": 0
            }

        try:
            search_results = self.keyword_search.search(query, limit)

            # Build local metadata lookup from config
            local_metadata = {d["id"]: d for d in self.config.get("datasets", [])}

            # Enrich results with local metadata where available
            enriched_results = []
            for r in search_results:
                dataset_id = r["dataset_id"]
                meta = local_metadata.get(dataset_id, {})

                # Read sync_info if available for row_count hint
                sync_info = self._read_sync_info(dataset_id)

                enriched_results.append({
                    "dataset_id": dataset_id,
                    "title": meta.get("title") or meta.get("name") or dataset_id,
                    "description": meta.get("description", ""),
                    "row_count": sync_info.get("row_count") if sync_info else None,
                    "columns": [],  # Not available in local index
                    "score": r["score"]
                })

            self._log_query(
                f"search_datasets(q={query}, limit={limit})",
                success=True,
                rows_returned=len(enriched_results)
            )

            return {
                "query": query,
                "results": enriched_results,
                "total": len(enriched_results)
            }

        except Exception as e:
            self._log_query(
                f"search_datasets(q={query}, limit={limit})",
                success=False,
                error_message=str(e)
            )
            return {
                "error": f"Search failed: {str(e)}",
                "query": query,
                "results": [],
                "total": 0
            }

    def _read_sync_info(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Read sync_info.json for a dataset if it exists."""
        sync_info_path = self.data_path / dataset_id / "sync_info.json"
        if sync_info_path.exists():
            try:
                with open(sync_info_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return None