"""
Keyword search using pre-built SQLite FTS5 index from R2.

Downloads the index on first use and provides fast keyword search
matching the remote API behavior.
"""

import os
import re
import sqlite3
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)

# R2 configuration
R2_BUCKET = "datasets"
R2_INDEX_PATH = "search/index_keywords.db"
R2_ENDPOINT = "https://{account_id}.r2.cloudflarestorage.com"


class KeywordSearchIndex:
    """Keyword search using SQLite FTS5 index downloaded from R2."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.index_path = base_path / "search" / "index_keywords.db"
        self._conn: Optional[sqlite3.Connection] = None

    def _get_r2_credentials(self) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Get R2 credentials from environment."""
        return (
            os.getenv("R2_ACCOUNT_ID"),
            os.getenv("R2_ACCESS_KEY_ID"),
            os.getenv("R2_SECRET_ACCESS_KEY")
        )

    def _download_from_api(self) -> bool:
        """Download keyword index from Subsets API."""
        api_url = os.getenv("SUBSETS_API_URL", "https://api.subsets.io")
        api_key = os.getenv("SUBSETS_API_KEY")

        try:
            import httpx

            logger.info(f"Downloading keyword index from API: {api_url}/search/index")

            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            with httpx.Client(timeout=60.0) as client:
                response = client.get(f"{api_url}/search/index", headers=headers)
                response.raise_for_status()

                # Write to file
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.index_path, "wb") as f:
                    f.write(response.content)

                size_mb = self.index_path.stat().st_size / 1024 / 1024
                logger.info(f"Downloaded keyword index from API ({size_mb:.2f} MB)")
                return True

        except ImportError:
            logger.info("httpx not available for API download")
            return False
        except Exception as e:
            logger.warning(f"API download failed: {e}")
            return False

    def _download_from_r2(self) -> bool:
        """Download keyword index from R2 (fallback)."""
        account_id, access_key, secret_key = self._get_r2_credentials()

        if not all([account_id, access_key, secret_key]):
            logger.warning("R2 credentials not configured")
            return False

        # Create directory
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        endpoint = R2_ENDPOINT.format(account_id=account_id)

        # Try boto3 first
        try:
            import boto3
            from botocore.config import Config

            logger.info("Downloading keyword index from R2 via boto3...")

            s3_client = boto3.client(
                's3',
                endpoint_url=endpoint,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                config=Config(signature_version='s3v4')
            )

            s3_client.download_file(R2_BUCKET, R2_INDEX_PATH, str(self.index_path))

            size_mb = self.index_path.stat().st_size / 1024 / 1024
            logger.info(f"Downloaded keyword index ({size_mb:.2f} MB)")
            return True

        except ImportError:
            logger.info("boto3 not available, trying CLI tools...")
        except Exception as e:
            logger.warning(f"boto3 download failed: {e}, trying CLI tools...")

        # Fallback to s5cmd
        s3_path = f"s3://{R2_BUCKET}/{R2_INDEX_PATH}"
        env = os.environ.copy()
        env["AWS_ACCESS_KEY_ID"] = access_key
        env["AWS_SECRET_ACCESS_KEY"] = secret_key

        try:
            result = subprocess.run(
                ["s5cmd", "--endpoint-url", endpoint, "cp", s3_path, str(self.index_path)],
                env=env,
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                size_mb = self.index_path.stat().st_size / 1024 / 1024
                logger.info(f"Downloaded keyword index via s5cmd ({size_mb:.2f} MB)")
                return True
            else:
                logger.warning(f"s5cmd failed: {result.stderr}")

        except FileNotFoundError:
            logger.info("s5cmd not available")
        except Exception as e:
            logger.warning(f"s5cmd error: {e}")

        return False

    def _download_index(self) -> bool:
        """Download keyword index - tries API first, then R2."""
        # Try API first (no credentials needed for public endpoint)
        if self._download_from_api():
            return True

        # Fallback to R2 direct download
        return self._download_from_r2()

    def ensure_index(self) -> bool:
        """Ensure the index is available, downloading if necessary."""
        if self.index_path.exists():
            return True
        return self._download_index()

    def _get_connection(self) -> Optional[sqlite3.Connection]:
        """Get SQLite connection, downloading index if needed."""
        if self._conn is not None:
            return self._conn

        if not self.ensure_index():
            return None

        try:
            self._conn = sqlite3.connect(self.index_path, check_same_thread=False)
            self._conn.execute("PRAGMA mmap_size = 268435456")  # 256MB mmap
            self._conn.execute("PRAGMA query_only = ON")
            logger.info(f"Opened keyword search index: {self.index_path}")
            return self._conn
        except Exception as e:
            logger.error(f"Failed to open index: {e}")
            return None

    @staticmethod
    def normalize(s: str) -> str:
        """Normalize string for matching."""
        return re.sub(r'[^a-z0-9]', '', s.lower())

    def search(self, query: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search datasets by keywords.

        Returns list of {dataset_id, score} dicts matching remote API format.
        """
        conn = self._get_connection()
        if conn is None:
            return []

        keywords = query.lower().split()
        if not keywords:
            return []

        # Score datasets by keyword matches
        dataset_scores = defaultdict(lambda: {"matches": set(), "count": 0})

        for i, kw in enumerate(keywords):
            kw_normalized = self.normalize(kw)
            if not kw_normalized:
                continue

            try:
                # FTS5 search - uses dataset_id column
                cursor = conn.execute("""
                    SELECT DISTINCT d.dataset_id
                    FROM keywords_fts f
                    JOIN dataset_keywords d ON f.rowid = d.rowid
                    WHERE keywords_fts MATCH ?
                """, (f'"{kw_normalized}"',))

                for (dataset_id,) in cursor:
                    dataset_scores[dataset_id]["matches"].add(i)
                    dataset_scores[dataset_id]["count"] += 1

            except sqlite3.OperationalError as e:
                logger.warning(f"FTS query failed for '{kw}': {e}")
                continue

        # Rank by keywords matched, then by total matches
        ranked = sorted(
            dataset_scores.items(),
            key=lambda x: (-len(x[1]["matches"]), -x[1]["count"], x[0])
        )

        return [
            {
                "dataset_id": dataset_id,
                "score": len(scores["matches"]) / len(keywords)
            }
            for dataset_id, scores in ranked[:limit]
        ]

    def search_with_titles(self, query: str, limit: int = 50) -> Dict[str, Any]:
        """
        Search and return results with titles.

        Matches the /search/with-titles API response format.
        """
        results = self.search(query, limit)

        # For local mode, title = dataset_id (we don't have titles in the index)
        return {
            "query": query,
            "results": [
                {
                    "dataset_id": r["dataset_id"],
                    "title": r["dataset_id"],  # Use ID as title
                    "score": r["score"]
                }
                for r in results
            ],
            "total": len(results)
        }

    def close(self):
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# Singleton instance
_search_index: Optional[KeywordSearchIndex] = None


def get_keyword_search(base_path: Optional[Path] = None) -> Optional[KeywordSearchIndex]:
    """Get or create keyword search index singleton."""
    global _search_index
    if _search_index is None:
        path = base_path or (Path.home() / "subsets")
        _search_index = KeywordSearchIndex(path)
    return _search_index
