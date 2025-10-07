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
from typing import Any, Dict, List, Optional, Sequence, Tuple
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

class WhitelistConfig:
    """Holds whitelist configuration for write operations and method calls."""

    def __init__(self, allowed_models: Optional[set[str]] = None, allowed_methods: Optional[set[Tuple[str, str]]] = None):
        # Default to deny-all
        self.allowed_models: set[str] = allowed_models or set()
        self.allowed_methods: set[Tuple[str, str]] = allowed_methods or set()

    @staticmethod
    def _parse_csv_list(value: str) -> list[str]:
        if not value:
            return []
        parts: list[str] = []
        for chunk in value.split(','):
            token = chunk.strip()
            if token:
                parts.append(token)
        return parts

    @classmethod
    def from_env(cls) -> "WhitelistConfig":
        """Build a whitelist from environment variables.

        Supported environment variables:
          - ODOO_WRITE_ALLOWED_MODELS
          - ODOO_ALLOWED_METHODS

        Methods accept tokens in either model.method or model:method format.
        """
        models_env = os.getenv("ODOO_WRITE_ALLOWED_MODELS", "")
        methods_env = os.getenv("ODOO_ALLOWED_METHODS", "")

        model_set = set(cls._parse_csv_list(models_env))

        method_set: set[Tuple[str, str]] = set()
        for token in cls._parse_csv_list(methods_env):
            if "." in token:
                model, method = token.split(".", 1)
            elif ":" in token:
                model, method = token.split(":", 1)
            else:
                logger.warning(f"Ignoring invalid method token '{token}', expected model.method")
                continue
            model = model.strip()
            method = method.strip()
            if model and method:
                method_set.add((model, method))

        cfg = cls(model_set, method_set)
        logger.info(
            "Whitelist loaded: %d write-allowed models, %d allowed methods",
            len(cfg.allowed_models), len(cfg.allowed_methods)
        )
        return cfg

class OdooConnector:
    """Handles connection and operations with Odoo database."""

    def __init__(self, url: str, db: str, username: str, password: str, whitelist: Optional[WhitelistConfig] = None):
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
        self.whitelist = whitelist or WhitelistConfig()

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

    def _validate_write_model(self, model_name: str) -> bool:
        """Validate if a model is allowed for write operations.

        Args:
            model_name: Name of the Odoo model

        Returns:
            True if model is whitelisted for writes

        Raises:
            ValueError: If model is not allowed for write operations
        """
        if model_name not in self.whitelist.allowed_models:
            raise ValueError(
                f"Write operations not allowed for model '{model_name}'. "
                f"Allowed models: {', '.join(sorted(self.whitelist.allowed_models))}"
            )
        return True

    def _validate_method_call(self, model_name: str, method_name: str) -> bool:
        """Validate if a method call is allowed.

        Args:
            model_name: Name of the Odoo model
            method_name: Name of the method

        Returns:
            True if method call is whitelisted

        Raises:
            ValueError: If method call is not allowed
        """
        method_key = (model_name, method_name)
        if method_key not in self.whitelist.allowed_methods:
            allowed_for_model = [method for model, method in self.whitelist.allowed_methods if model == model_name]
            if allowed_for_model:
                raise ValueError(
                    f"Method '{method_name}' not allowed for model '{model_name}'. "
                    f"Allowed methods for this model: {', '.join(allowed_for_model)}"
                )
            else:
                raise ValueError(
                    f"No methods allowed for model '{model_name}'. "
                    f"Model not in whitelist or no methods configured."
                )
        return True

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

    def create_record(
        self,
        model_name: str,
        values: Dict[str, Any]
    ) -> int:
        """Create a new record.

        Args:
            model_name: Name of the Odoo model
            values: Dictionary of field values

        Returns:
            ID of the created record
        """
        if not self.ensure_connected():
            raise RuntimeError("Not connected to Odoo")

        try:
            # Validate model is allowed for write operations
            self._validate_write_model(model_name)

            if model_name not in self.odoo.env:
                raise ValueError(f"Model '{model_name}' does not exist")

            record_id = self.odoo.env[model_name].create(values)
            logger.info(f"Created record {record_id} in model '{model_name}'")
            return record_id

        except Exception as e:
            logger.error(f"Error creating record in model '{model_name}': {e}")
            raise

    def write_records(
        self,
        model_name: str,
        record_ids: List[int],
        values: Dict[str, Any]
    ) -> bool:
        """Update existing records.

        Args:
            model_name: Name of the Odoo model
            record_ids: List of record IDs to update
            values: Dictionary of field values to update

        Returns:
            True if successful
        """
        if not self.ensure_connected():
            raise RuntimeError("Not connected to Odoo")

        try:
            # Validate model is allowed for write operations
            self._validate_write_model(model_name)

            if model_name not in self.odoo.env:
                raise ValueError(f"Model '{model_name}' does not exist")

            if not record_ids:
                return True

            self.odoo.env[model_name].write(record_ids, values)
            logger.info(f"Updated {len(record_ids)} records in model '{model_name}'")
            return True

        except Exception as e:
            logger.error(f"Error updating records in model '{model_name}': {e}")
            raise

    def call_method(
        self,
        model_name: str,
        method_name: str,
        record_ids: List[int] = None,
        args: List[Any] = None,
        kwargs: Dict[str, Any] = None
    ) -> Any:
        """Call a method on a model or recordset.

        Args:
            model_name: Name of the Odoo model
            method_name: Name of the method to call
            record_ids: List of record IDs (for recordset methods)
            args: Positional arguments for the method
            kwargs: Keyword arguments for the method

        Returns:
            Method result
        """
        if not self.ensure_connected():
            raise RuntimeError("Not connected to Odoo")

        try:
            # Validate method call is allowed
            self._validate_method_call(model_name, method_name)

            if model_name not in self.odoo.env:
                raise ValueError(f"Model '{model_name}' does not exist")

            model = self.odoo.env[model_name]

            # If record_ids provided, call method on recordset
            if record_ids:
                recordset = model.browse(record_ids)
                method = getattr(recordset, method_name)
            else:
                # Call method on model
                method = getattr(model, method_name)

            # Prepare arguments
            args = args or []
            kwargs = kwargs or {}

            # Call the method
            result = method(*args, **kwargs)
            logger.info(f"Called method '{method_name}' on model '{model_name}'")
            return result

        except Exception as e:
            logger.error(f"Error calling method '{method_name}' on model '{model_name}': {e}")
            raise

# Initialize the MCP server
mcp = FastMCP("Odoo")

# Global connector instance (singleton for the MCP server lifecycle)
connector: Optional[OdooConnector] = None

def get_connector() -> OdooConnector:
    """Get or create the Odoo connector instance."""
    global connector

    if connector is None:
        # Lazily initialize with env-based whitelist if not explicitly created in serve()
        url = os.getenv("ODOO_URL", "").strip()
        db = os.getenv("ODOO_DB", "").strip()
        username = os.getenv("ODOO_USERNAME", "").strip()
        password = os.getenv("ODOO_PASSWORD", "").strip()

        if not all([url, db, username, password]):
            raise RuntimeError(
                "Missing Odoo connection details. Please set ODOO_URL, ODOO_DB, "
                "ODOO_USERNAME, and ODOO_PASSWORD environment variables or create a .env file."
            )

        wl = WhitelistConfig.from_env()
        connector = OdooConnector(url, db, username, password, whitelist=wl)

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


@mcp.tool()
def create_odoo_record(
    model_name: str,
    values: Dict[str, Any]
) -> int:
    """Create a new record in an Odoo model.

    Args:
        model_name: The technical name of the Odoo model
        values: Dictionary of field values for the new record

    Returns the ID of the newly created record. Use this to create new records
    such as messages, tasks, or other data in Odoo.

    Note: Only whitelisted models are allowed for security. Use list_write_whitelist()
    to see which models are permitted, or validate_write_operation() to check before calling.
    """
    try:
        conn = get_connector()
        record_id = conn.create_record(model_name=model_name, values=values)
        return record_id
    except Exception as e:
        raise RuntimeError(f"Failed to create record in model '{model_name}': {e}")


@mcp.tool()
def update_odoo_records(
    model_name: str,
    record_ids: List[int],
    values: Dict[str, Any]
) -> bool:
    """Update existing records in an Odoo model.

    Args:
        model_name: The technical name of the Odoo model
        record_ids: List of record IDs to update
        values: Dictionary of field values to update

    Returns True if the update was successful. Use this to modify existing
    records in Odoo.

    Note: Only whitelisted models are allowed for security. Use list_write_whitelist()
    to see which models are permitted, or validate_write_operation() to check before calling.
    """
    try:
        conn = get_connector()
        success = conn.write_records(
            model_name=model_name,
            record_ids=record_ids,
            values=values
        )
        return success
    except Exception as e:
        raise RuntimeError(f"Failed to update records in model '{model_name}': {e}")


@mcp.tool()
def call_odoo_method(
    model_name: str,
    method_name: str,
    record_ids: List[int] = None,
    args: List[Any] = None,
    kwargs: Dict[str, Any] = None
) -> Any:
    """Call a method on an Odoo model or recordset.

    Args:
        model_name: The technical name of the Odoo model
        method_name: Name of the method to call
        record_ids: List of record IDs (for recordset methods). If None, calls on model
        args: Positional arguments for the method
        kwargs: Keyword arguments for the method

    Returns the result of the method call. Use this to call specific Odoo methods
    like message_post, action methods, or custom business logic.

    Note: Only whitelisted model/method combinations are allowed for security.
    """
    try:
        conn = get_connector()
        result = conn.call_method(
            model_name=model_name,
            method_name=method_name,
            record_ids=record_ids,
            args=args,
            kwargs=kwargs
        )
        return result
    except Exception as e:
        raise RuntimeError(f"Failed to call method '{method_name}' on model '{model_name}': {e}")


@mcp.tool()
def list_write_whitelist() -> Dict[str, Any]:
    """List the current whitelist configuration for write operations.

    Returns a dictionary containing:
    - write_allowed_models: List of models allowed for create/write operations
    - allowed_methods: List of (model, method) combinations allowed for method calls

    Use this to check what operations are currently permitted.
    """
    conn = get_connector()
    return {
        "write_allowed_models": sorted(list(conn.whitelist.allowed_models)),
        "allowed_methods": sorted([f"{model}.{method}" for model, method in conn.whitelist.allowed_methods])
    }


@mcp.tool()
def validate_write_operation(
    operation_type: str,
    model_name: str,
    method_name: str = None
) -> Dict[str, Any]:
    """Validate if a write operation would be allowed without executing it.

    Args:
        operation_type: Type of operation ('create', 'write', or 'method_call')
        model_name: The technical name of the Odoo model
        method_name: Name of the method (required for 'method_call' operations)

    Returns a dictionary with validation results:
    - allowed: Boolean indicating if operation is allowed
    - reason: Explanation of why operation is allowed/denied
    - suggestions: List of alternative allowed operations if applicable
    """
    try:
        conn = get_connector()

        result = {
            "allowed": False,
            "reason": "",
            "suggestions": []
        }

        if operation_type in ['create', 'write']:
            try:
                conn._validate_write_model(model_name)
                result["allowed"] = True
                result["reason"] = f"Model '{model_name}' is whitelisted for {operation_type} operations"
            except ValueError as e:
                result["reason"] = str(e)
                result["suggestions"] = sorted(list(conn.whitelist.allowed_models))

        elif operation_type == 'method_call':
            if not method_name:
                result["reason"] = "method_name is required for method_call validation"
                return result

            try:
                conn._validate_method_call(model_name, method_name)
                result["allowed"] = True
                result["reason"] = f"Method '{method_name}' is allowed on model '{model_name}'"
            except ValueError as e:
                result["reason"] = str(e)
                # Get allowed methods for this model
                allowed_for_model = [method for model, method in conn.whitelist.allowed_methods if model == model_name]
                if allowed_for_model:
                    result["suggestions"] = allowed_for_model
                else:
                    result["suggestions"] = [f"{model}.{method}" for model, method in conn.whitelist.allowed_methods]

        else:
            result["reason"] = f"Unknown operation_type '{operation_type}'. Must be 'create', 'write', or 'method_call'"

        return result

    except Exception as e:
        return {
            "allowed": False,
            "reason": f"Validation error: {e}",
            "suggestions": []
        }


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

    # Initialize connector with whitelist from environment (deny-all if unset)
    url = os.getenv("ODOO_URL", "").strip()
    db = os.getenv("ODOO_DB", "").strip()
    username = os.getenv("ODOO_USERNAME", "").strip()
    password = os.getenv("ODOO_PASSWORD", "").strip()

    wl = WhitelistConfig.from_env()

    global connector
    connector = OdooConnector(url, db, username, password, whitelist=wl)

    # Test connection at startup
    try:
        if connector.connect():
            logger.info("Successfully connected to Odoo")
        else:
            logger.error("Failed to connect to Odoo at startup")
            return
    except Exception as e:
        logger.error(f"Failed to initialize Odoo connection: {e}")
        return

    # Run the MCP server
    mcp.run()

