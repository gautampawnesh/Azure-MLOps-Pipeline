from azureml.core import Workspace, Datastore, Dataset

# 1. Connect to your Workspace
ws = Workspace.from_config() # Assumes you have a config.json in the folder

# 2. Get the Datastore you created in the Portal
datastore = Datastore.get(ws, 'workspaceblobstore')

# 3. Upload the files from the VM to the Datastore
datastore.upload(src_dir='/home/pgautam2/data/raw',
                 target_path='raw_data/',
                 overwrite=True,
                 show_progress=True)

# 4. Optional: Register it as a FileDataset so the Pipeline can "see" it
ds = Dataset.File.from_files(path=(datastore, 'raw_data/'))
ds.register(workspace=ws, name='biometric_raw_data', description='Iris and Fingerprint BMPs')
