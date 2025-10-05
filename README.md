# Odoo MCP Server

A Model Context Protocol (MCP) server that provides LLMs with read-only access to Odoo databases. This server exposes Odoo models and records through the MCP protocol, allowing AI assistants to query and analyze your Odoo data.

## Features

- **Read-only access** to Odoo databases via OdooRPC
- **Model discovery** - List all available Odoo models
- **Field introspection** - Get detailed field information for any model
- **Flexible querying** - Search records with Odoo's domain syntax
- **Record reading** - Fetch specific records by ID
- **Count operations** - Get record counts without fetching data
- **Environment-based configuration** - Secure credential management
- **MCP compliance** - Works with any MCP-compatible client

## Installation

Install directly with uvx (recommended):

# Odoo MCP Server

A Model Context Protocol (MCP) server that provides LLMs with read-only access to Odoo databases. This server exposes Odoo models and records through the MCP protocol, allowing AI assistants to query and analyze your Odoo data.

## Features

- Read-only access to Odoo databases via OdooRPC
- Model discovery – list all available Odoo models
- Field introspection – get detailed field information for any model
- Flexible querying – search records with Odoo's domain syntax
- Record reading – fetch specific records by ID
- Count operations – get record counts without fetching data
- Environment-based configuration – secure credential management
- MCP compliance – works with any MCP-compatible client

## Installation

Install directly with uvx (recommended):

```bash
uvx odoorpc-mcp
```

## VS Code MCP setup

Create a `.vscode/mcp.json` in your workspace with:

```jsonc
{
	"servers": {
		"odoo": {
			"type": "stdio",
			"command": "uvx",
			"args": [
				"odoorpc-mcp"
			],
			"env": {
				"ODOO_URL": "${input:odoo_url}",
				"ODOO_DB": "${input:odoo_db}",
				"ODOO_USERNAME": "${input:odoo_username}",
				"ODOO_PASSWORD": "${input:odoo_password}"
			}
		}
	},
	"inputs": [
		{
			"id": "odoo_url",
			"type": "promptString",
			"description": "Odoo Server URL (e.g., https://your-odoo.com)"
		},
		{
			"id": "odoo_db",
			"type": "promptString",
			"description": "Odoo Database Name"
		},
		{
			"id": "odoo_username",
			"type": "promptString",
			"description": "Odoo Username"
		},
		{
			"id": "odoo_password",
			"type": "promptString",
			"description": "Odoo Password",
			"password": true
		}
	]
}
```

Alternatively, put credentials in a `.env` file in your workspace:

```bash
ODOO_URL=https://your-odoo-instance.com
ODOO_DB=your_database
ODOO_USERNAME=your_username
ODOO_PASSWORD=your_password
```

## Usage

- Show config example:

```bash
odoorpc-mcp --config-example
```

- Print version:

```bash
odoorpc-mcp --version
```

- Start the MCP server (requires environment variables):

```bash
odoorpc-mcp
```

## Development

Build locally:

```bash
uv build
```

Run from source:

```bash
PYTHONPATH=src python3 -m odoo_mcp.cli --config-example
```

## License

MIT

