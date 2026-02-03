# Subsets MCP Server

Query statistical datasets with AI assistants or from the command line.

## Quick Start

### Claude Desktop

Add to your config:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "subsets": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/subsetsio/subsets-mcp-server.git",
        "mcp-server", "--api-key", "YOUR_API_KEY"
      ]
    }
  }
}
```

Get your API key at [subsets.io/settings](https://subsets.io/settings). Requires [uv](https://docs.astral.sh/uv/).

### Claude Code

```bash
claude mcp add subsets -- uvx --from git+https://github.com/subsetsio/subsets-mcp-server.git mcp-server --api-key YOUR_API_KEY
```

### CLI Installation

```bash
# Install the CLI
pip install git+https://github.com/subsetsio/subsets-mcp-server.git

# Log in with your API key
subsets login

# Search for datasets
subsets search "gdp growth"

# Add a dataset to your local collection
subsets add wdi_gdp_per_capita

# Query with local DuckDB
subsets query "SELECT * FROM wdi_gdp_per_capita LIMIT 10"
```

---

## How It Works

Subsets runs **locally** on your machine:

1. **Search** the remote catalog to discover datasets
2. **Add** datasets to download them to your local collection
3. **Query** with DuckDB - 100% local compute, works offline

Your data stays on your machine. Subsets indexes public data; it does not own it.

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_datasets` | Search the remote Subsets catalog |
| `inspect_datasets` | Get detailed info, stats, and preview for datasets |
| `list_datasets` | List your locally installed datasets |
| `add_datasets` | Download datasets to your local collection |
| `remove_datasets` | Remove datasets from your local collection |
| `sync_datasets` | Update installed datasets to latest versions |
| `execute_query` | Run SQL queries on local datasets |

### search_datasets

Search the Subsets catalog for datasets.

```python
search_datasets(query="gdp growth", limit=20)
```

- `query` (required): Search keywords
- `limit` (optional): Max results (1-100, default 20)

### inspect_datasets

Get detailed information, statistics, and preview for datasets.

```python
inspect_datasets(dataset_ids=["wdi_gdp_per_capita", "wdi_population"])
```

- `dataset_ids` (required): List of dataset IDs (1-20)

Returns metadata, schema, column statistics, preview rows, and sync status.

### list_datasets

List datasets in your local collection.

```python
list_datasets(query="gdp")  # Optional filter
```

- `query` (optional): Filter by keyword

### add_datasets

Download datasets to your local collection.

```python
add_datasets(dataset_ids=["wdi_gdp_per_capita", "wdi_population"])
```

- `dataset_ids` (required): List of dataset IDs to download

### remove_datasets

Remove datasets from your local collection.

```python
remove_datasets(dataset_ids=["wdi_gdp_per_capita"])
```

- `dataset_ids` (required): List of dataset IDs to remove

### sync_datasets

Update installed datasets to their latest versions.

```python
sync_datasets()  # Sync all
sync_datasets(dataset_ids=["wdi_gdp_per_capita"])  # Sync specific
```

- `dataset_ids` (optional): List of dataset IDs to sync. If omitted, syncs all.

### execute_query

Execute a SQL query against your locally installed datasets.

```python
execute_query(query="SELECT * FROM wdi_gdp_per_capita LIMIT 10", output_format="tsv")
```

- `query` (required): SQL query to execute
- `output_format` (optional): `json` or `tsv` (default)

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `subsets login` | Log in with your API key |
| `subsets logout` | Log out |
| `subsets search <query>` | Search the catalog |
| `subsets inspect <ids...>` | Get dataset details |
| `subsets list [query]` | List installed datasets |
| `subsets add <ids...>` | Download datasets |
| `subsets remove <ids...>` | Remove datasets |
| `subsets sync [ids...]` | Update datasets |
| `subsets query <sql>` | Execute SQL |
| `subsets status` | Show current status |

### Examples

```bash
# Search for datasets
subsets search "unemployment rate"

# Inspect multiple datasets
subsets inspect wdi_gdp_per_capita wdi_population

# Add multiple datasets at once
subsets add wdi_gdp_per_capita wdi_population eurostat_unemployment

# Add datasets from a file
subsets add --file datasets.txt

# List installed datasets
subsets list

# Filter installed datasets
subsets list gdp

# Query with SQL
subsets query "SELECT country, year, value FROM wdi_gdp_per_capita WHERE year = 2020"

# Output as JSON
subsets query "SELECT * FROM wdi_gdp_per_capita LIMIT 10" --format json

# Sync all datasets
subsets sync

# Sync specific datasets
subsets sync wdi_gdp_per_capita wdi_population

# Remove datasets
subsets remove wdi_gdp_per_capita

# Check status
subsets status
```

---

## Configuration

Config is stored in `~/subsets/config.json`. Data is stored in `~/subsets/data/`.

---

## License

MIT
