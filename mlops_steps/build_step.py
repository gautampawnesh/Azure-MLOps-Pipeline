import subprocess
from step_utils import * # TODO
from azureml.core import Model, Run
from environs import Env
import click
import sys
import importlib
import ast
import configparser
import logging
from pathlib import Path
from network import subprocess_with_retry
from naming import *
from azureml.core import Model, Run
from environs import Env
from logging_init import setup_logging

CONFIG_FILE_DUMP_PATH = './config_after_step_execution.py'

_logger = logging.getLogger(__name__)

@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.option(
    '--feature_repo_url',
    type=str,
    help='URL to feature repo in the following form: '
    'git+https://<ACCESS_TOKEN>@<URL_TO_REPO>@<COMMIT_SHA>#egg=<PYTHON_PACKAGE_NAME>',
)
@click.option(
    '--python_entrypoint_module', type=str, help='module name for the code entrypoint'
)
@click.option(
    '--python_entrypoint_function',
    type=str,
    help='function name for script',
)
@click.option(
    '--ml_config_module',
    type=str,
    help='Module path (i.e. without .py, e.g. '
    '\'biometric.configs.experiment1\')',
)
@click.option(
    '--env_relative_filepath',
    type=str,
    help='The relative file path to the .env file seen from'
    'feature repository root folder.',
)
@click.option(
    '--output_dir',
    type=str,
    help='Str path to the folder where output of the component is saved.',
)
@click.option(
    '--input_dirs',
    type=str,
    multiple=True,
    nargs=2,
    help='Two strings: <name of input folder> <folder path>. This '
    'argumented can be given multiple times to support multiple input folders.',
)
@click.option(
    "--model_id",
    type=str,
    help="model_name:model_version for some model in the Azure model registry "
    "which should be mounted to the step. Relative path to the actual checkpoint must "
    "be specified in the mmlab_config using resume-from / load-from.",
)
@click.option(
    '--description', type=str, help='Some description text that is saved to Azure ML.'
)
def build_mlops_step(
    feature_repo_url,
    python_entrypoint_module,
    python_entrypoint_function,
    ml_config_module,
    env_relative_filepath,
    output_dir=None,
    input_dirs=None,
    description=None,
    model_id=None,
):
    """This function is executed in each AzureML step of the pipeline."""
    _logger.info(
        f'Starting step: {python_entrypoint_module}, {ml_config_module}, {env_relative_filepath}'
    )
    step_run = Run.get_context()
    pipeline_run = step_run.parent

    # install the feature repository using pip and git url
    subprocess_with_retry(check_output=False,
                          arg_list=[
                              sys.executable,
                              '-m',
                              'pip',
                              'install',
                              '--no-deps',
                              '--force-reinstall',
                              '--no-cache-dir',
                              feature_repo_url,
                          ],
                          log_information=feature_repo_url,
                          )
    # save the commit of the feature repo which is used for the step (should
    # be identical in all steps as we replace the branch by commit before
    # entering this function)
    step_run.set_tags(
        {
            'feature_repo_commit': get_branch_or_commit(feature_repo_url),
            'feature_repo_url': feature_repo_url,
            'description': description,
        }
    )
    pipeline_run.set_tags(
        {
            'feature_repo_commit': get_branch_or_commit(feature_repo_url),
            'feature_repo_url': feature_repo_url,
            'description': description,
        }
    )

    try:
        _logger.info(subprocess.check_output(['nvidia-smi']).decode('utf-8'))
        _logger.info(subprocess.check_output(['nvcc', '--version']).decode('utf-8'))
    except FileNotFoundError:
        pass  # on CPU compute instances
    
    # set environment from file
    feature_package_name = get_feature_package_name(feature_repo_url)
    env_filepath = (
            Path(importlib.import_module(feature_package_name).__file__).parent.parent
            / env_relative_filepath
    )
    _logger.info(f'Env filepath is: {env_filepath}')
    env = Env()
    env.read_env(env_filepath)

    entrypoint_module = importlib.import_module(python_entrypoint_module)
    _logger.info(f'Imported entrypoint module: {python_entrypoint_module}')

    if output_dir is not None:
        # we need to create this folder for some cases, otherwise path cannot be found
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    if model_id:
        # Load model from azure registry
        pass
    
    if input_dirs:
        return_list = getattr(entrypoint_module, python_entrypoint_function)(
            ml_config_module, env, output_dir, {name: path for name, path in input_dirs}
        )
    else:
        return_list = getattr(entrypoint_module, python_entrypoint_function)(
            ml_config_module, env, output_dir
        )
    
    log_targets = [pipeline_run, step_run]
    if return_list is not None:
        # it is important that processing happens in the order of the elements
        # in return_list (for upload_folder / register_model after training) # TODO
        for entry in return_list:
            if entry['logfunc_azure'] is not None:
                for run in log_targets:
                    if 'azure_kwargs' in entry:
                        getattr(run, entry['logfunc_azure'])(
                            entry['key'], entry['value'], **entry['azure_kwargs']
                        )
                    else:
                        getattr(run, entry['logfunc_azure'])(
                            entry['key'], entry['value']
                        )


if __name__ == '__main__':
    setup_logging()
    build_mlops_step()
