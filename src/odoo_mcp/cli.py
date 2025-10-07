#!/usr/bin/env python3
"""
CLI interface for the Odoo MCP Server
"""

import argparse
import os
from . import __version__


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OdooRPC MCP Server - Bridge between LLMs and Odoo via Model Context Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  odoorpc-mcp                  # Start the MCP server
  odoorpc-mcp --help           # Show this help message
  
Environment Variables:
  ODOO_URL        Odoo server URL (e.g., https://your-odoo-instance.com)
  ODOO_DB         Database name
  ODOO_USERNAME   Username  
  ODOO_PASSWORD   Password

Configuration:
  You can also create a .env file in your current directory with these variables.
        """
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"odoorpc-mcp {__version__}"
    )
    
    parser.add_argument(
        "--config-example",
        action="store_true",
        help="Show example VS Code MCP configuration"
    )
    
    # Whitelist controls (comma-separated)
    parser.add_argument(
        "--write-allowed-models",
        dest="write_allowed_models",
        help="Comma-separated list of models allowed for create/write operations (overrides env)",
    )
    parser.add_argument(
        "--allowed-methods",
        dest="allowed_methods",
        help="Comma-separated list of model.method allowed calls (overrides env)",
    )

    args = parser.parse_args()

    if args.config_example:
        print_config_example()
        return

    # Start the MCP server
    # Lazy import to avoid importing server module when only showing help/version
    from . import server

    # If CLI provided overrides, set env vars before serve() loads them
    if args.write_allowed_models:
        os.environ["ODOO_WRITE_ALLOWED_MODELS"] = args.write_allowed_models
    if args.allowed_methods:
        os.environ["ODOO_ALLOWED_METHODS"] = args.allowed_methods

    server.serve()


def print_config_example():
    """Print example VS Code configuration."""
    config = '''
VS Code MCP Configuration Example (.vscode/mcp.json):

Create a file at .vscode/mcp.json with the following content:

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

Optional: instead of prompts, you can use a .env file alongside your workspace:

ODOO_URL=https://your-odoo-instance.com
ODOO_DB=your_database_name
ODOO_USERNAME=your_username
ODOO_PASSWORD=your_password
'''
    print(config)


if __name__ == "__main__":
    main()