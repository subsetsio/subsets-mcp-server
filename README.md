# Subsets MCP Server

Model Context Protocol server for querying statistical datasets with AI assistants.

## Quick Start

### Using Claude Desktop

Add to your Claude Desktop config:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "subsets": {
      "command": "uv",
      "args": [
        "run",
        "https://github.com/subsetsio/subsets-mcp-server",
        "--api-key",
        "YOUR_API_KEY"
      ]
    }
  }
}
```

Get your API key at [subsets.io/settings](https://subsets.io/settings)

Requires [uv](https://docs.astral.sh/uv/) installed.

### Using Other MCP Clients

```bash
uv run https://github.com/subsetsio/subsets-mcp-server --api-key YOUR_API_KEY
```

## Available MCP Tools

### `list_datasets`
Search and browse available datasets with semantic search.

**Parameters:**
- `q` (string): Search query
- `limit` (integer): Max results (default: 10)
- `min_score` (float): Min relevance score threshold (0.0-2.0)

**Example:**
```
list_datasets(q="unemployment europe", limit=5)
```

### `get_dataset`
Get detailed information about a specific dataset including schema, statistics, and preview.

**Parameters:**
- `dataset_id` (string): Dataset identifier

**Returns:** Full metadata, column descriptions, row counts, data preview, and query usage stats

**Example:**
```
get_dataset("eurostat_unemployment_2024")
```

### `execute_sql_query`
Run SQL queries on datasets using DuckDB.

**Parameters:**
- `query` (string): SQL SELECT statement

**Example:**
```sql
execute_sql_query("SELECT * FROM eurostat_unemployment_2024 LIMIT 10")
```

## CLI Tools

The package also includes CLI commands for managing local datasets:

```bash
# Install globally
npm install -g @subsetsio/mcp-server

# Add datasets to local collection
subsets add eurostat_unemployment_2024

# Download datasets
subsets sync

# List local datasets
subsets list

# View status
subsets status
```

## Local Development

```bash
# Clone repository
git clone https://github.com/subsetsio/subsets-mcp-server
cd subsets-mcp-server

# Install dependencies
uv sync

# Run MCP server
uv run python src/server.py --api-key YOUR_API_KEY

# Or run CLI
uv run python src/cli.py --help
```

## Documentation

Full documentation at [subsets.io/docs](https://subsets.io/docs)

## License

MIT
