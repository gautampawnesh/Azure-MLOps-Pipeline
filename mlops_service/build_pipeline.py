
import logging
from datetime import datetime
from pathlib import Path

import click
import configparser
from azureml.core import Dataset, Datastore
from azureml.core.compute import AmlCompute
from azureml.core.runconfig import RunConfiguration
from azureml.data.datapath import DataPath
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.exceptions import UserErrorException
from azureml.data.output_dataset_config import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from msrest.exceptions import HttpOperationError
from mlops_service.pipeline_utils import *
import mlops_steps
from mlops_steps.logging_init import setup_logging
from mlops_steps.naming import *

_logger = logging.getLogger(__name__)


@click.command()
@click.option('--ml_config_module', type=str, help='Train Config Module')
@click.option(
    '--env_relative_filepath',
    type=str,
    help='Path to env file from feature repo root '
         '(e.g. biometric-recognition/configs/.env.four_v100)',
)
@click.option(
    '--feature_repo_url',
    type=str,
    help='URL to feature repo in the following form: '
         'git+https://<ACCESS_TOKEN>@<URL_TO_REPO>@<BRANCH_OR_COMMIT>#egg=<PYTHON_PACKAGE_NAME>',
)
@click.option(
    '--model_id',
    type=str,
    help='model_name:model_version for some model in the Azure model registry '
         'which should be mounted to the step. Relative path to the actual checkpoint must '
         'be specified in the mmlab_config using resume-from / load-from.',
)
@click.option(
    '--description', type=str, help='Some description text that is saved to Azure ML.'
)
@click.option(
    '--run_preprocessing',
    type=str,
    help='Str True/False whether preprocessing step should be executed.',
)
@click.option(
    '--run_training',
    type=str,
    help='Str True/False whether training step should be executed.',
)
@click.option(
    '--run_evaluation',
    type=str,
    help='Str True/False whether evaluation step should be executed.',
)
@click.option(
    '--run_visualizer',
    type=str,
    help='Str True/False whether visualization step should be executed.',
)
def build_aml_pipeline(
        ml_config_module,
        env_relative_filepath,
        feature_repo_url,
        model_id,
        description,
        run_preprocessing,
        run_training,
        run_evaluation,
        run_visualizer):
    """
    AML Pipeline for training and evaluation for any feature repos.
    """
    run_preprocessing = eval_boolean_str(run_preprocessing)
    run_training = eval_boolean_str(run_training)
    run_evaluation = eval_boolean_str(run_evaluation)
    run_visualizer = eval_boolean_str(run_visualizer)

    experiment_name = f'{get_feature_package_name(feature_repo_url)}-training-evaluation'
    # Download the repo 
    feature_download_path = Path(__file__).parent.parent
    download_feature_repo(feature_repo_url=feature_repo_url,
                          target_path=feature_download_path)

    # Read environment and configuration from feature repo.
    (
        env,
        aml_runconfig_gpu,
        aml_runconfig_cpu,
    ) = read_config_and_env(
        feature_package_folder=feature_download_path,
        env_relative_filepath=env_relative_filepath
    )

    def get_step(step_name,
                 compute_target,
                 runconfig,
                 entrypoint_module,
                 entrypoint_function,
                 aml_inputs,
                 aml_outputs,
                 **kwargs):
        """return PythonScriptStep"""
        # default (required) kwargs
        karg_list = [
            '--feature_repo_url', feature_repo_url,
            '--python_entrypoint_module', entrypoint_module,
            '--ml_config_module', ml_config_module,
            '--python_entrypoint_function', entrypoint_function,
            '--env_relative_filepath', env_relative_filepath,
            '--description', description,
        ]
        # a list of outputs?
        if aml_outputs:
            karg_list.extend(['--output_dir', aml_outputs[0]])

        aml_inputs_lst = []
        for name, aml_input in aml_inputs.items():
            karg_list.extend(['--input_dirs', name, aml_input])
            aml_inputs_lst.append(aml_input)

        # extra kwargs
        for k, v in kwargs.items():
            if v is not None:
                karg_list.extend([f'--{k}', v])

        _logger.info(f'Creation step {step_name}')
        _logger.info(f'Compute target {compute_target}')
        _logger.info(f'Src Directory {str(Path(mlops_steps.__file__).parent)}')
        _logger.info(f'Inputs {aml_inputs}')
        _logger.info(f'Ouputs {aml_outputs}')
        _logger.info(f'Arguments {karg_list} (with kwargs {kwargs})')
        _logger.info(f'Run Config {runconfig}')
        _logger.info(f'Entry point module {entrypoint_module}, function {entrypoint_function}')

        step = PythonScriptStep(
            name=step_name,
            script_name="mlops_steps/build_step.py",
            compute_target=compute_target,
            source_directory=str(Path(__file__).parent.parent),
            inputs=aml_inputs_lst,
            outputs=aml_outputs,
            arguments=karg_list,
            runconfig=runconfig,
            allow_reuse=True)
        return step

    # read environment parameters
    intermediate_datastore_name = env('INTERMEDIATE_DATASTORE_NAME')

    # Connect to the AML workspace
    aml_ws = get_workspace(env)
    _logger.info(f'Connected to workspace: {aml_ws}')

    # Get Azure machine learning clusters
    aml_compute_gpu = get_compute_target(env('AML_GPU_CLUSTER_NAME'), aml_ws)
    _logger.info(f'AML training compute {aml_compute_gpu}')
    aml_compute_cpu = get_compute_target(
        env('AML_CPU_CLUSTER_NAME'), aml_ws
    )
    _logger.info(f'AML preprocessing compute {aml_compute_cpu}')

    # configure data stores
    try:
        intermediate_data_store = Datastore.get(aml_ws, intermediate_datastore_name)
    except HttpOperationError as err:
        _logger.error(f'Got HttpOperationError {err} exception on data store ')
        raise err
    except Exception as err:
        _logger.error(f'Got exception on data store: {err}')
        raise err
    config = configparser.ConfigParser()
    config.read(ml_config_module)
    #import pdb; pdb.set_trace()
    datastore_name = config['MLOPS']['data_store_name']
    path_on_datastore = config['MLOPS']['path_on_datastore']
    mount_path = config['MLOPS']['mount_path']
    training_output = config['MLOPS']['training_output']
    evaluation_output = config['MLOPS']['evaluation_output']

    raw_inputs_dirs = dict()

    try:
        input_data_store = Datastore.get(aml_ws, datastore_name)
    except HttpOperationError as err:
        _logger.error(f'Got HttpOperationError {err} exception on data store ')
        raise err
    else:
        raw_inputs_dir = DatasetConsumptionConfig(
            "raw_inputs",
            Dataset.File.from_files(
                DataPath(datastore=input_data_store,
                        path_on_datastore=path_on_datastore),
                validate=False
            ),
            mode='mount',
            path_on_compute=mount_path,
        )
        raw_inputs_dirs["raw_inputs"] = raw_inputs_dir
        _logger.info(f"Mounted : {str(input_data_store.as_mount())}")

    steps = []
    if run_preprocessing:
        preprocessing_output_path = PipelineData(
            'preprocessing_output', datastore=intermediate_data_store
        )
        preprocessing_step = get_step(
            step_name='preprocessing',
            #compute_target=aml_compute_gpu,
            compute_target=aml_compute_cpu,
            runconfig=aml_runconfig_cpu,
            entrypoint_module=env('PREPROCESSING_PYTHON_ENTRYPOINT_MODULE'),
            entrypoint_function=env('PREPROCESSING_PYTHON_ENTRYPOINT_FUNCTION'),
            aml_inputs=raw_inputs_dirs,
            aml_outputs=[preprocessing_output_path],
            model_id=model_id,
        )
        steps.append(preprocessing_step)
    else:
        preprocessing_output_path = None
    
    if run_training:
        train_output_path = PipelineData(
            'training_output', datastore=intermediate_data_store
        )
        training_step = get_step(
            step_name='training',
            #compute_target=aml_compute_gpu,
            compute_target=aml_compute_cpu,
            runconfig=aml_runconfig_cpu,
            entrypoint_module=env('TRAIN_PYTHON_ENTRYPOINT_MODULE'),
            entrypoint_function=env('TRAIN_PYTHON_ENTRYPOINT_FUNCTION'),
            aml_inputs=raw_inputs_dirs,
            aml_outputs=[train_output_path],
            model_id=model_id,
        )
        steps.append(training_step)
    else:
        train_output_path = None

    # evaluation step
    if run_evaluation and run_training:
        evaluation_output_path = PipelineData(
            'evaluation_output', datastore=intermediate_data_store
        )
        aml_inputs_evaluation = raw_inputs_dirs
        aml_inputs_evaluation["training_output"] = train_output_path

        evaluation_step = get_step(
            step_name='evaluation',
            #compute_target=aml_compute_gpu,
            compute_target=aml_compute_cpu,
            runconfig=aml_runconfig_cpu,
            entrypoint_module=env('EVALUATION_PYTHON_ENTRYPOINT_MODULE'),
            entrypoint_function=env('EVALUATION_PYTHON_ENTRYPOINT_FUNCTION'),
            aml_inputs=aml_inputs_evaluation,
            aml_outputs=[evaluation_output_path],
            model_id=model_id,
        )
        steps.append(evaluation_step)
    else:
        evaluation_output_path = None


    # register_step = get_step(
    #     step_name="register",
    #     compute_target=aml_compute_cpu,
    #     runconfig=aml_runconfig_gpu,
    #     entrypoint_module=env("REGISTER_PYTHON_ENTRYPOINT_MODULE"),
    #     entrypoints_function=env("REGISTER_PYTHON_ENTRYPOINT_FUNCTION"),
    #     aml_inputs="",
    #     aml_outputs=""
    # )
    # steps.append(register_step)

    # build, publish and submit the pipeline
    train_pipeline = Pipeline(workspace=aml_ws, steps=steps)
    train_pipeline.validate()
    _logger.info(f'Built pipeline {train_pipeline}')
    _logger.info(f'model id {model_id}')

    # publish the pipeline
    published_pipeline = train_pipeline.publish(
        name=f'{experiment_name}-pipeline',
        description=description,
        version=datetime.utcnow().isoformat(),
    )
    _logger.info(
        f'Published pipeline {published_pipeline.name},'
        f' version {published_pipeline.version},'
        f' endpoint: {published_pipeline.endpoint}'
    )

    run = published_pipeline.submit(aml_ws, experiment_name)
    run.wait_for_completion()


if __name__ == "__main__":
    setup_logging()
    build_aml_pipeline()
