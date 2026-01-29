"""Azure Key Vault utilities for securely retrieving secrets.

This module provides a unified interface for retrieving secrets, supporting both:
1. Local development: Uses OPENROUTER_API_KEY environment variable directly
2. AML/Cloud: Retrieves secrets from Azure Key Vault using managed identity

Usage:
    from Magmathor.Utils.keyvault import get_secret

    # This will automatically use the appropriate method:
    # - If OPENROUTER_API_KEY is set, returns it directly
    # - Otherwise, retrieves from Azure Key Vault using managed identity
    api_key = get_secret("openrouter-api-key")
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


# Environment variable names for Key Vault configuration
KEYVAULT_NAME_ENV = "AZURE_KEYVAULT_NAME"
KEYVAULT_URL_ENV = "AZURE_KEYVAULT_URL"

# Mapping of secret names to their environment variable equivalents
# This allows local development to use env vars while cloud uses Key Vault
SECRET_TO_ENV_MAP = {
    "openrouter-api-key": "OPENROUTER_API_KEY",
    "openai-api-key": "OPENAI_API_KEY",
}


def _get_keyvault_url() -> str | None:
    """Get the Key Vault URL from environment variables.

    Supports both direct URL (AZURE_KEYVAULT_URL) and vault name (AZURE_KEYVAULT_NAME).
    """
    # Check for direct URL first
    url = os.getenv(KEYVAULT_URL_ENV)
    if url:
        return url

    # Fall back to constructing URL from vault name
    vault_name = os.getenv(KEYVAULT_NAME_ENV)
    if vault_name:
        return f"https://{vault_name}.vault.azure.net/"

    return None


@lru_cache(maxsize=32)
def _get_secret_from_keyvault(secret_name: str) -> str:
    """Retrieve a secret from Azure Key Vault using managed identity.

    This function is cached to avoid repeated Key Vault calls for the same secret.

    Args:
        secret_name: The name of the secret in Key Vault

    Returns:
        The secret value

    Raises:
        ValueError: If Key Vault is not configured
        Exception: If secret retrieval fails
    """
    vault_url = _get_keyvault_url()
    if not vault_url:
        raise ValueError(
            f"Azure Key Vault not configured. Set {KEYVAULT_NAME_ENV} or {KEYVAULT_URL_ENV} environment variable."
        )

    try:
        # Import lazily to avoid dependency issues when running locally
        from azure.identity import DefaultAzureCredential
        from azure.keyvault.secrets import SecretClient

        logger.info(f"Retrieving secret '{secret_name}' from Key Vault: {vault_url}")

        # DefaultAzureCredential will automatically use:
        # - Managed Identity when running on AML
        # - Azure CLI credentials when running locally (if logged in)
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)

        secret = client.get_secret(secret_name)
        logger.info(f"Successfully retrieved secret '{secret_name}'")
        return secret.value

    except ImportError as e:
        raise ImportError(
            "Azure Key Vault SDK not installed. "
            "Install with: pip install azure-identity azure-keyvault-secrets"
        ) from e


def get_secret(
    secret_name: str,
    *,
    use_keyvault: bool | None = None,
    default: str | None = None,
) -> str | None:
    """Retrieve a secret, automatically choosing the appropriate source.

    Priority order:
    1. If the corresponding environment variable is set, use it directly
    2. If Key Vault is configured, retrieve from Key Vault
    3. Return the default value (or None)

    Args:
        secret_name: The name of the secret (Key Vault name, e.g., "openrouter-api-key")
        use_keyvault: Force Key Vault usage (True) or env var (False).
                      If None (default), auto-detect based on what's available.
        default: Default value if secret is not found

    Returns:
        The secret value, or default if not found

    Raises:
        ValueError: If secret is required but not found anywhere
    """
    # Get the corresponding environment variable name
    env_var_name = SECRET_TO_ENV_MAP.get(
        secret_name, secret_name.upper().replace("-", "_")
    )

    # Check environment variable first (unless forced to use Key Vault)
    if use_keyvault is not True:
        env_value = os.getenv(env_var_name)
        if env_value:
            logger.debug(
                f"Using secret '{secret_name}' from environment variable {env_var_name}"
            )
            return env_value

    # Try Key Vault if configured (unless forced to use env var)
    if use_keyvault is not False:
        vault_url = _get_keyvault_url()
        if vault_url:
            try:
                return _get_secret_from_keyvault(secret_name)
            except Exception as e:
                logger.warning(f"Failed to retrieve secret from Key Vault: {e}")
                if use_keyvault is True:
                    # If explicitly requested Key Vault, raise the error
                    raise

    # Return default or None
    if default is not None:
        return default

    return None


def get_openrouter_api_key() -> str:
    """Convenience function to get the OpenRouter API key.

    Returns:
        The OpenRouter API key

    Raises:
        ValueError: If the API key is not found in any source
    """
    api_key = get_secret("openrouter-api-key")
    if not api_key:
        raise ValueError(
            "OpenRouter API key not found. Either:\n"
            "1. Set the OPENROUTER_API_KEY environment variable, or\n"
            f"2. Set {KEYVAULT_NAME_ENV} and store the key as 'openrouter-api-key' in Key Vault"
        )
    return api_key


def is_running_on_aml() -> bool:
    """Check if currently running on Azure ML.

    Returns:
        True if running on AML, False otherwise
    """
    # Common AML environment indicators
    aml_indicators = [
        "AZUREML_RUN_ID",
        "AZUREML_EXPERIMENT_NAME",
        "AZUREML_WORKSPACE_NAME",
    ]
    return any(os.getenv(var) for var in aml_indicators)


def setup_secrets_for_aml() -> None:
    """Set up secrets from Key Vault as environment variables.

    This is useful for setting up the environment at the start of an AML job,
    so that code expecting environment variables will work without modification.
    """
    if not is_running_on_aml():
        logger.debug("Not running on AML, skipping Key Vault secret setup")
        return

    vault_url = _get_keyvault_url()
    if not vault_url:
        logger.warning("Running on AML but Key Vault not configured")
        return

    # Set up each mapped secret as an environment variable
    for secret_name, env_var in SECRET_TO_ENV_MAP.items():
        if not os.getenv(env_var):
            try:
                value = _get_secret_from_keyvault(secret_name)
                os.environ[env_var] = value
                logger.info(f"Set {env_var} from Key Vault secret '{secret_name}'")
            except Exception as e:
                logger.debug(f"Could not retrieve secret '{secret_name}': {e}")
