import os
import sys
import shutil
import logging
import tempfile
from environs import Env
from pathlib import Path
from azureml.core import Workspace
from azureml.core.authentication import (
    InteractiveLoginAuthentication,
    ServicePrincipalAuthentication,
)
from azureml.core.compute import AmlCompute
from azureml.core.runconfig import (
    CondaDependencies,  #TODO
    DockerConfiguration,
    RunConfiguration,
)
from azureml.exceptions import ComputeTargetException
from dotenv.main import dotenv_values
from mlops_steps.network import subprocess_with_retry

_logger = logging.getLogger(__name__)


def eval_boolean_str(param):
    """Create a boolean from a string representing a boolean value."""
    if param is None:
        return False
    param = param.strip()
    if param in ['True', 'true']:
        param = True
    elif param in ['False', 'false']:
        param = False
    else:
        param = True
    return param

def _get_authentication_method(
    env):
    tenant_id = env("TENANT_ID")
    interactive_auth = env.bool("INTERACTIVE_AUTH")

    if interactive_auth:
        auth_method = InteractiveLoginAuthentication(tenant_id=tenant_id)
    else:
        service_principal_id = env("servicePrincipalId")
        service_principal_key = env("servicePrincipalKey")
        auth_method = ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=service_principal_id,
            service_principal_password=service_principal_key,
        )
    return auth_method


def get_workspace(env):
    subscription_id = env("SUBSCRIPTION_ID")
    resource_group = env("RESOURCE_GROUP")
    workspace_name = env("WORKSPACE_NAME")
    pat_token = env.str("PIP_PAT_TOKEN", "")
    pip_url = env.str("PIP_EXTRA_URL", "")

    aml_workspace = Workspace(
        subscription_id,
        resource_group,
        workspace_name,
        auth=_get_authentication_method(env),
    )
    if pip_url:
        info = f"Adding connection to python feed {pip_url}"
        extra_info = (
            " with Personal Access Token (PAT)"
            if pat_token
            else " without Authentication, 'PIP_PAT_TOKEN' environment variable not found."
        )
        _logger.info(info + extra_info)

        aml_workspace.set_connection(
            name=f"{workspace_name}-PythonFeed",
            category="PythonFeed",
            target=pip_url,
            authType="PAT" if pat_token else "",
            value=pat_token,
        )
    return aml_workspace


def download_feature_repo(feature_repo_url, target_path):
    """
    Args:
        feature_repo_url: URL to the feature repo of the form
            "git+https://<ACCESS_TOKEN>@<URL_TO_REPO>@<COMMIT_SHA>#egg=<PYTHON_PACKAGE_NAME>"
        target_path: the feature repo will be copied
    example:
        python -m pip download git+https://github_pat@github.com/gautampawnesh/Multimodal-Biometric-Recognition-System@main#egg=biometric-recognition
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        feature_package_name = feature_repo_url.split("#egg=")[-1]
        _logger.info(f"Feature package name: {feature_package_name}")

        subprocess_with_retry(
            check_output=False,
            arg_list=[
                sys.executable,
                "-m",
                "pip",
                "download",
                "-d",
                tmpdirname,
                "--no-deps",
                "--no-binary=:all:",
                feature_repo_url,
            ],
            log_information=f"downloading feature repo {feature_repo_url}",
        )
        files = Path(tmpdirname).glob("*.*")
        last = max(files, key=lambda x: x.stat().st_ctime)
        shutil.unpack_archive(last, tmpdirname)
        # we only want to copy the python package code into the Azure snapshot, not
        # the whole repository, which might be too big # TODO
        shutil.copytree(
            Path(tmpdirname) / feature_package_name / feature_package_name,
            target_path / feature_package_name
        )
        _logger.info(f"extracted feature repo: {os.listdir(tmpdirname)}")


def read_config_and_env(feature_package_folder, env_relative_filepath):
    """Read Env and config"""
    env_filepath = feature_package_folder / env_relative_filepath
    _logger.info(f"Env file path: {env_filepath}")

    env = Env()
    env.read_env(env_filepath)

    # get runtime environments
    aml_runconfig_gpu = create_run_configuration(
        env,
        conda_dependencies_filepath=str(
            feature_package_folder / env("GPU_DEPENDENCIES_FILE")
        ),
        dockerfile_filepath=str(feature_package_folder / env("DOCKERFILE_GPU")),
    )
    aml_runconfig_cpu = create_run_configuration(
        env,
        conda_dependencies_filepath=str(
            feature_package_folder / env("CPU_DEPENDENCIES_FILE")
        ),
        dockerfile_filepath=str(feature_package_folder / env("DOCKERFILE_CPU")),
    )

    return (
        env,
        aml_runconfig_gpu,
        aml_runconfig_cpu,
    )


def get_compute_target(compute_target_name, aml_ws):
    """Either return the AmlCompute defined by compute_target_name,
    otherwise log an error."""
    try:
        aml_compute = AmlCompute(aml_ws, compute_target_name)
        _logger.info(f"found existing compute target {compute_target_name}")
        return aml_compute
    except ComputeTargetException:
        _logger.error(
            "Target not found, existing targets in workspace are", exc_info=True
        )
        for ct in aml_ws.compute_targets:
            _logger.info(ct)


def create_run_configuration(
    env,
    conda_dependencies_filepath,
    dockerfile_filepath,
    docker_enabled= True,
):
    # Choose the python dependencies TODO
    pip_url = env.str("PIP_EXTRA_URL", "")
    conda_dependencies = CondaDependencies(
        conda_dependencies_file_path=conda_dependencies_filepath
    )
    if pip_url:
        conda_dependencies.set_pip_option(f"--extra-index-url {pip_url}")
    run_config = RunConfiguration(conda_dependencies=conda_dependencies)
    # run_config.environment_variables["AZUREML_COMPUTE_USE_COMMON_RUNTIME"] = "false"
    # Increase shared memory
    docker_runtime_config = DockerConfiguration(
        use_docker=True, shm_size="32g", arguments=["--rm"]
    )
    run_config.docker = docker_runtime_config

    # Choose dockerfile
    # run_config.environment.docker.enabled = docker_enabled
    # run_config.environment.docker.arguments = ["--shm-size=32g", "--rm"]
    run_config.environment.docker.base_image = None
    with open(dockerfile_filepath) as f:
        dockerfile = f.read()
    run_config.environment.docker.base_dockerfile = dockerfile

    run_config.environment.python.user_managed_dependencies = False
    run_config.environment.environment_variables = _merge_environments()

    return run_config


def _merge_environments():
    dot_env = dotenv_values()
    local_env = os.environ
    keys_dot_env = set(dot_env.keys())
    keys_local_env = set(local_env.keys())
    intersection_keys = keys_dot_env & keys_local_env
    intersection_dict = {key: os.getenv(key) for key in intersection_keys}
    return {**dot_env, **intersection_dict}