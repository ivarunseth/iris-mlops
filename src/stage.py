"""
MLflow model staging and promotion utilities.

This module provides helper functions to:
- Wait for a model version to reach READY state.
- Assign or delete aliases (e.g., 'staging', 'production') for registered models.
- Promote the latest version through staging to production.
- Demote models from production back to staging, or remove aliases entirely.

The CLI supports two actions:
    promote → Promote latest model version (None → staging → production)
    demote  → Demote latest model version (production → staging → remove)
"""

import time
import argparse

import mlflow

from settings import Settings


def wait_until_ready(client, name, version, timeout_s=120):
    """
    Wait until a specific model version reaches READY status.

    Args:
        client (MlflowClient): The MLflow client instance.
        name (str): Registered model name.
        version (str | int): Model version number.
        timeout_s (int): Timeout in seconds before raising an error.

    Raises:
        TimeoutError: If the model version does not become READY in time.
    """
    start = time.time()
    while time.time() - start < timeout_s:
        mv = client.get_model_version(name=name, version=version)
        if mv.status == "READY":
            return
        time.sleep(1.5)
    raise TimeoutError(f"Model version {name} v{version} not READY within {timeout_s}s")


def set_alias(alias, settings=Settings(), version=None):
    """
    Assign a registered model alias to a specific version.

    If no version is provided, the latest version is used.

    Args:
        alias (str): Alias name (e.g., 'staging', 'production').
        settings (Settings): Project settings instance.
        version (str | int, optional): Model version to assign alias to.

    Returns:
        str: Version number that the alias was set to.
    """
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()
    model_name = settings.registered_model_name

    if version is None:
        versions = client.search_model_versions(f"name = '{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        latest = max(versions, key=lambda v: int(v.version))
        version = latest.version

    wait_until_ready(client, model_name, version)
    client.set_registered_model_alias(
        name=model_name, alias=alias.lower(), version=version
    )
    return version


def delete_alias(alias, settings=Settings()):
    """
    Remove a registered model alias.

    Args:
        alias (str): Alias name to delete.
        settings (Settings): Project settings instance.
    """
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()
    model_name = settings.registered_model_name
    client.delete_registered_model_alias(model_name, alias.lower())


def promote_latest_auto(settings=Settings()):
    """
    Promote the latest model version through staging → production.

    Logic:
        - If latest version has no alias → assign 'staging'
        - If latest version is in staging → promote to 'production'
        - If already in production → raise an error
    """
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()
    model_name = settings.registered_model_name

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'")
    latest = max(versions, key=lambda v: int(v.version))

    aliases = client.get_registered_model(model_name).aliases or {}
    alias_for_latest = [a for a, v in aliases.items() if str(v) == str(latest.version)]

    if not alias_for_latest:
        next_alias = "staging"
    elif "staging" in alias_for_latest:
        next_alias = "production"
    else:
        raise ValueError("Already in production or unknown alias")

    v = set_alias(next_alias, settings=settings, version=latest.version)
    print(f"Promoted version {v} of {model_name} to alias '{next_alias}'")


def demote_latest_auto(settings=Settings()):
    """
    Demote the latest version from production → staging, or staging → no alias.

    Priority:
        1. If latest is in production, remove 'production'.
        2. Else if in staging, remove 'staging'.
        3. Else raise error.
    """
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()
    model_name = settings.registered_model_name

    aliases = client.get_registered_model(model_name).aliases or {}
    reverse_aliases = {int(v): a for a, v in aliases.items()}  # version → alias

    if not reverse_aliases:
        raise ValueError(f"No aliases found for model '{model_name}'")

    latest_version = max(reverse_aliases.keys())

    if "production" in aliases:
        delete_alias("production", settings=settings)
        print(f"Removed alias 'production' from version {latest_version} of {model_name}")
        return

    if "staging" in aliases:
        delete_alias("staging", settings=settings)
        print(f"Removed alias 'staging' from version {latest_version} of {model_name}")
        return

    raise ValueError(f"No production or staging alias to demote for version {latest_version}")


def main():
    """Command-line interface for promoting or demoting MLflow model versions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["promote", "demote"])
    args = parser.parse_args()

    if args.action == "promote":
        promote_latest_auto()
    elif args.action == "demote":
        demote_latest_auto()


if __name__ == "__main__":
    main()
