#!/usr/bin/env python3
"""
Odoo MCP Server

An MCP (Model Context Protocol) server that exposes Odoo database records to LLMs.
This server acts as a bridge between LLMs and Odoo, allowing for dynamic queries
and model introspection.

Usage:
    odoo-mcp

Environment Variables:
    ODOO_URL: Odoo server URL (e.g., https://your-odoo-instance.com)
    ODOO_DB: Database name
    ODOO_USERNAME: Username
    ODOO_PASSWORD: Password

The server can also read these from a .env file.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import odoorpc
from dotenv import load_dotenv
from fastmcp import FastMCP

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class OdooConnector:
    """Handles connection and operations with Odoo database."""
    
    def __init__(self, url: str, db: str, username: str, password: str):
        """Initialize the Odoo connector.
        
        Args:
            url: Odoo server URL
            db: Database name
            username: Username
            password: Password
        """
        self.db = db
        self.username = username
        self.password = password
        
        # Parse URL to extract host, protocol, and port
        parsed_url = urlparse(url if url.startswith(('http://', 'https://')) else f'https://{url}')
        self.host = parsed_url.hostname
        self.protocol = 'jsonrpc+ssl' if parsed_url.scheme == 'https' else 'jsonrpc'
        self.port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
        
        self.odoo: Optional[odoorpc.ODOO] = None
        self._connected = False
        
        logger.info(f"Initialized connector for {self.host}:{self.port} ({self.protocol})")
    
    def connect(self) -> bool:
        """Connect to Odoo server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to Odoo at {self.host}:{self.port}")
            
            self.odoo = odoorpc.ODOO(
                host=self.host,
                protocol=self.protocol,
                port=self.port
            )
            
            self.odoo.login(
                db=self.db,
                login=self.username,
                password=self.password
            )
            
            self._connected = True
            logger.info(f"Successfully connected to Odoo (version: {self.odoo.version})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Odoo: {e}")
            self._connected = False
            return False
    
    def ensure_connected(self) -> bool:
        """Ensure connection is active, reconnect if necessary.
        
        Returns:
            True if connected, False otherwise
        """
        if not self._connected or not self.odoo:
            return self.connect()
        
        # Test connection with a simple operation
        try:
            _ = self.odoo.env.uid
            return True
        except Exception:
            logger.warning("Connection lost, attempting to reconnect")
            return self.connect()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models in the Odoo database.
        
        Returns:
            List of models with their information
        """
        if not self.ensure_connected():
            raise RuntimeError("Not connected to Odoo")
        
        try:
            ir_model = self.odoo.env['ir.model']
            model_ids = ir_model.search([])
            models = ir_model.read(model_ids, ['model', 'name', 'info'])
            
            logger.info(f"Retrieved {len(models)} models")
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise
    
    def get_model_fields(self, model_name: str) -> Dict[str, Any]:
        """Get field information for a specific model.
        
        Args:
            model_name: Name of the Odoo model
            
        Returns:
            Dictionary of field information
        """
        if not self.ensure_connected():
            raise RuntimeError("Not connected to Odoo")
        
        try:
            if model_name not in self.odoo.env:
                raise ValueError(f"Model '{model_name}' does not exist")
            
            fields_info = self.odoo.env[model_name].fields_get()
            logger.info(f"Retrieved {len(fields_info)} fields for model '{model_name}'")
            return fields_info
            
        except Exception as e:
            logger.error(f"Error getting fields for model '{model_name}': {e}")
            raise
    
    def search_records(
        self,
        model_name: str,
        domain: List[Any] = None,
        fields: List[str] = None,
        limit: int = 100,
        offset: int = 0,
        order: str = None
    ) -> List[Dict[str, Any]]:
        """Search records in a model using search_read.
        
        Args:
            model_name: Name of the Odoo model
            domain: Search domain (list of tuples)
            fields: Fields to retrieve
            limit: Maximum number of records
            offset: Number of records to skip
            order: Sort order
            
        Returns:
            List of records
        """
        if not self.ensure_connected():
            raise RuntimeError("Not connected to Odoo")
        
        try:
            if model_name not in self.odoo.env:
                raise ValueError(f"Model '{model_name}' does not exist")
            
            domain = domain or []
            
            records = self.odoo.env[model_name].search_read(
                domain=domain,
                fields=fields,
                limit=limit,
                offset=offset,
                order=order
            )
            
            logger.info(f"Retrieved {len(records)} records from model '{model_name}'")
            return records
            
        except Exception as e:
            logger.error(f"Error searching records in model '{model_name}': {e}")
            raise
    
    def read_records(
        self,
        model_name: str,
        record_ids: List[int],
        fields: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Read specific records by ID.
        
        Args:
            model_name: Name of the Odoo model
            record_ids: List of record IDs
            fields: Fields to retrieve
            
        Returns:
            List of records
        """
        if not self.ensure_connected():
            raise RuntimeError("Not connected to Odoo")
        
        try:
            if model_name not in self.odoo.env:
                raise ValueError(f"Model '{model_name}' does not exist")
            
            if not record_ids:
                return []
            
            records = self.odoo.env[model_name].read(record_ids, fields)
            logger.info(f"Read {len(records)} records from model '{model_name}'")
            return records
            
        except Exception as e:
            logger.error(f"Error reading records from model '{model_name}': {e}")
            raise
    
    def count_records(self, model_name: str, domain: List[Any] = None) -> int:
        """Count records matching the domain.
        
        Args:
            model_name: Name of the Odoo model
            domain: Search domain
            
        Returns:
            Number of matching records
        """
        if not self.ensure_connected():
            raise RuntimeError("Not connected to Odoo")
        
        try:
            if model_name not in self.odoo.env:
                raise ValueError(f"Model '{model_name}' does not exist")
            
            domain = domain or []
            count = self.odoo.env[model_name].search_count(domain)
            logger.info(f"Counted {count} records in model '{model_name}'")
            return count
            
        except Exception as e:
            logger.error(f"Error counting records in model '{model_name}': {e}")
            raise

# Initialize the MCP server
mcp = FastMCP("Odoo")

# Global connector instance
connector: Optional[OdooConnector] = None

def get_connector() -> OdooConnector:
    """Get or create the Odoo connector instance."""
    global connector
    
    if connector is None:
        # Get connection details from environment
        url = os.getenv("ODOO_URL", "").strip()
        db = os.getenv("ODOO_DB", "").strip()
        username = os.getenv("ODOO_USERNAME", "").strip()
        password = os.getenv("ODOO_PASSWORD", "").strip()
        
        if not all([url, db, username, password]):
            raise RuntimeError(
                "Missing Odoo connection details. Please set ODOO_URL, ODOO_DB, "
                "ODOO_USERNAME, and ODOO_PASSWORD environment variables or create a .env file."
            )
        
        connector = OdooConnector(url, db, username, password)
    
    return connector


def validate_environment() -> bool:
    """Check if all required environment variables are set."""
    required_vars = ["ODOO_URL", "ODOO_DB", "ODOO_USERNAME", "ODOO_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables or create a .env file with the required values.")
        logger.error("Run 'odoorpc-mcp --config-example' for setup instructions.")
        return False
    
    return True

@mcp.tool()
def list_odoo_models() -> List[Dict[str, Any]]:
    """List all available models in the Odoo database.
    
    Returns a list of models with their technical names, display names, and descriptions.
    Use this to discover what data is available in the Odoo instance.
    """
    try:
        conn = get_connector()
        models = conn.list_models()
        
        # Sort by model name for better readability
        models.sort(key=lambda x: x.get('model', ''))
        
        return models
    except Exception as e:
        raise RuntimeError(f"Failed to list models: {e}")


@mcp.tool()
def get_model_fields(model_name: str) -> Dict[str, Any]:
    """Get detailed field information for a specific Odoo model.
    
    Args:
        model_name: The technical name of the Odoo model (e.g., 'res.partner', 'sale.order')
    
    Returns detailed information about each field including type, string label, help text,
    required status, and other metadata. Use this to understand the structure of a model
    before querying its data.
    """
    try:
        conn = get_connector()
        fields = conn.get_model_fields(model_name)
        return fields
    except Exception as e:
        raise RuntimeError(f"Failed to get fields for model '{model_name}': {e}")


@mcp.tool()
def search_odoo_records(
    model_name: str,
    domain: List[Any] = None,
    fields: List[str] = None,
    limit: int = 100,
    offset: int = 0,
    order: str = None
) -> List[Dict[str, Any]]:
    """Search for records in an Odoo model.
    
    Args:
        model_name: The technical name of the Odoo model (e.g., 'res.partner', 'sale.order')
        domain: Search criteria as a list of tuples, e.g., [('name', 'ilike', 'John'), ('is_company', '=', False)]
        fields: List of field names to retrieve. If None, gets all fields
        limit: Maximum number of records to return (default: 100)
        offset: Number of records to skip (for pagination)
        order: Sort order, e.g., 'name asc' or 'create_date desc'
    
    Returns a list of records matching the search criteria. The domain parameter uses
    Odoo's domain syntax with tuples of (field, operator, value).
    
    Common operators: '=', '!=', '<', '>', '<=', '>=', 'in', 'not in', 'ilike', 'like'
    """
    try:
        conn = get_connector()
        records = conn.search_records(
            model_name=model_name,
            domain=domain,
            fields=fields,
            limit=limit,
            offset=offset,
            order=order
        )
        return records
    except Exception as e:
        raise RuntimeError(f"Failed to search records in model '{model_name}': {e}")


@mcp.tool()
def read_odoo_records(
    model_name: str,
    record_ids: List[int],
    fields: List[str] = None
) -> List[Dict[str, Any]]:
    """Read specific records by their IDs.
    
    Args:
        model_name: The technical name of the Odoo model
        record_ids: List of record IDs to read
        fields: List of field names to retrieve. If None, gets all fields
    
    Returns the full record data for the specified IDs. This is more efficient than
    search when you know the exact record IDs you need.
    """
    try:
        conn = get_connector()
        records = conn.read_records(
            model_name=model_name,
            record_ids=record_ids,
            fields=fields
        )
        return records
    except Exception as e:
        raise RuntimeError(f"Failed to read records from model '{model_name}': {e}")


@mcp.tool()
def count_odoo_records(
    model_name: str,
    domain: List[Any] = None
) -> int:
    """Count records matching the search criteria.
    
    Args:
        model_name: The technical name of the Odoo model
        domain: Search criteria as a list of tuples (same format as search_odoo_records)
    
    Returns the number of records that match the domain criteria. This is useful for
    pagination or getting statistics without retrieving the actual records.
    """
    try:
        conn = get_connector()
        count = conn.count_records(model_name=model_name, domain=domain)
        return count
    except Exception as e:
        raise RuntimeError(f"Failed to count records in model '{model_name}': {e}")


@mcp.resource("odoo://models")
def get_models_resource() -> str:
    """Get a summary of all available Odoo models as a resource."""
    try:
        conn = get_connector()
        models = conn.list_models()
        
        # Create a formatted summary
        summary = "# Odoo Models Summary\n\n"
        summary += f"Total models available: {len(models)}\n\n"
        
        for model in sorted(models, key=lambda x: x.get('model', '')):
            model_name = model.get('model', 'Unknown')
            display_name = model.get('name', 'No name')
            info = model.get('info', 'No description available')
            
            summary += f"## {model_name}\n"
            summary += f"**Display Name:** {display_name}\n"
            if info:
                summary += f"**Description:** {info}\n"
            summary += "\n"
        
        return summary
        
    except Exception as e:
        return f"Error loading models: {e}"


def serve():
    """Start the MCP server after validating configuration."""
    # Validate configuration
    if not validate_environment():
        return
    
    logger.info("Starting OdooRPC MCP Server...")
    
    # Test connection at startup
    try:
        conn = get_connector()
        if conn.connect():
            logger.info("Successfully connected to Odoo")
        else:
            logger.error("Failed to connect to Odoo at startup")
            return
    except Exception as e:
        logger.error(f"Failed to initialize Odoo connection: {e}")
        return
    
    # Run the MCP server
    mcp.run()

