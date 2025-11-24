import os
from typing import Optional, Dict, Any
import requests


class RemoteDataPlatform:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
    
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
        """List available datasets from the remote Subsets data warehouse"""
        params = {
            "limit": limit,
            "offset": offset,
            "detailed": detailed
        }

        if license:
            params["license"] = license
        if git_user:
            params["git_user"] = git_user
        if q:
            params["q"] = q
        if min_score is not None:
            params["min_score"] = min_score
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            response = requests.get(
                f"{self.api_url}/datasets",
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Failed to list datasets: {str(e)}",
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific dataset including summary statistics"""
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }

        try:
            response = requests.get(
                f"{self.api_url}/datasets/{dataset_id}/summary",
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Failed to get dataset: {str(e)}",
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }

    def execute_sql_query(self, query: str, output_format: str = "json") -> Dict[str, Any]:
        """Execute a SQL query against remote datasets via the server API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use the server's SQL query endpoint
        payload = {
            "query": query
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/sql/query",
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()

            # Server returns: {columns: [...], rows: [[...], ...], row_count, execution_time_ms}
            if output_format == "tsv":
                # Check if result has the expected structure
                if "columns" in result and "rows" in result:
                    columns = result["columns"]
                    rows = result["rows"]

                    # Create TSV format
                    tsv_lines = []
                    # Header row
                    tsv_lines.append('\t'.join(columns))
                    # Data rows
                    for row in rows:
                        # Convert each value to string, handling None as empty string
                        row_str = '\t'.join(str(val) if val is not None else '' for val in row)
                        tsv_lines.append(row_str)

                    tsv_content = '\n'.join(tsv_lines)

                    return {
                        "format": "tsv",
                        "content": tsv_content,
                        "row_count": result.get("row_count", len(rows)),
                        "column_count": len(columns),
                        "render_hint": "markdown_table",
                        "instructions": "This TSV data should be rendered as a markdown table"
                    }

            # Return JSON format as-is
            return result
            
        except requests.exceptions.RequestException as e:
            error_detail = None
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                except:
                    error_detail = e.response.text
            
            return {
                "error": f"Failed to execute SQL query: {str(e)}",
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None,
                "detail": error_detail
            }