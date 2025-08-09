import argparse
import mlflow
import time

from settings import Settings


def wait_until_ready(client, name, version, timeout_s=120):
    """Wait until a model version is in READY status."""
    start = time.time()
    while time.time() - start < timeout_s:
        mv = client.get_model_version(name=name, version=version)
        if mv.status == "READY":
            return
        time.sleep(1.5)
    raise TimeoutError(f"Model version {name} v{version} not READY within {timeout_s}s")


def set_alias(alias, settings=Settings(), version=None):
    """Set an alias for a specific model version."""
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
    """Delete an alias from the registered model."""
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()
    model_name = settings.registered_model_name
    client.delete_registered_model_alias(model_name, alias.lower())


def promote_latest_auto(settings=Settings()):
    """Promote latest version through staging → production."""
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
        raise ValueError(f"Already in production or unknown alias")

    v = set_alias(next_alias, settings=settings, version=latest.version)
    print(f"Promoted version {v} of {model_name} to alias '{next_alias}'")


def demote_latest_auto(settings=Settings()):
    """Demote: production → staging, staging → remove alias."""
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()
    model_name = settings.registered_model_name

    aliases = client.get_registered_model(model_name).aliases or {}
    reverse_aliases = {int(v): a for a, v in aliases.items()}  # version → alias

    if not reverse_aliases:
        raise ValueError(f"No aliases found for model '{model_name}'")

    latest_version = max(reverse_aliases.keys())

    # Priority: remove production first if it exists
    if "production" in aliases:
        delete_alias("production", settings=settings)
        print(f"Removed alias 'production' from version {latest_version} of {model_name}")
        return

    # If production not present, then handle staging
    if "staging" in aliases:
        delete_alias("staging", settings=settings)
        print(f"Removed alias 'staging' from version {latest_version} of {model_name}")
        return

    raise ValueError(f"No production or staging alias to demote for version {latest_version}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["promote", "demote"])
    args = parser.parse_args()

    if args.action == "promote":
        promote_latest_auto()
    elif args.action == "demote":
        demote_latest_auto()


if __name__ == "__main__":
    main()
