import logging

from mlops_steps.network import subprocess_with_retry

_logger = logging.getLogger(__name__)


def get_repo_url_with_commit(feature_repo_url: str):
    """Return a git url that refers to a certain commit. If the original
    `feature_repo_url` referenced a commit already, this will be kept. If the
    `feature_repo_url` referenced a branch, the latest commit of this branch
    will be used. Needs internet connection to the feature repo.

    Args:
        feature_repo_url: URL to the feature repo of the form
            "git+https://<ACCESS_TOKEN>@<URL_TO_REPO>@<BRANCH_OR_COMMIT>#egg=<PYTHON_PACKAGE_NAME>"

    Returns:
        str: url like the feature_repo_url but with <BRANCH_OR_COMMIT> replaced by
            a commit in case that it was a branch

    """
    assert feature_repo_url.startswith("git+")
    url_split = feature_repo_url.split("@")
    assert len(url_split) == 3
    git_url = f"{url_split[0][4:]}@{url_split[1]}"
    branch_or_commit = get_branch_or_commit(feature_repo_url)
    package_name = get_feature_package_name(feature_repo_url)

    _logger.info(f"Getting commit for {git_url}")
    git_commit = subprocess_with_retry(check_output=True,
                    arg_list=["git", "ls-remote", git_url, branch_or_commit],
                    log_information=f"commit for {git_url}",
            ).decode("utf-8")
    # variable git_commit is empty str if the feature_repo_url contains a commit sha instead
    # of a branch name, hence we take the commit from the feature_repo_url in that case
    if not git_commit:
        git_commit = branch_or_commit
    else:
        # remove some additional part of the output
        git_commit = git_commit.split("\t")[0]
    _logger.info(f"Git commit of feature repo: {git_commit}")

    return f"{url_split[0]}@{url_split[1]}@{git_commit}#egg={package_name}"


def get_feature_package_name(feature_repo_url: str):
    """Return the feature package name.

    Args:
        feature_repo_url: URL to feature repo with #egg= in the end, specifying
                            the python package name

    Returns:
        str

    """
    return feature_repo_url.split("#egg=")[-1]


def get_branch_or_commit(feature_repo_url: str):
    return feature_repo_url.split("@")[-1].split("#egg=")[0]


def get_experiment_name(feature_repo_url: str):
    """
    Return the experiment name which is <feature_package_name>-<branch_name>.

    Args:
        feature_repo_url: URL to feature repo with #egg= in the end, specifying
                            the python package name

    Returns:
        str

    """
    feature_package_name = get_feature_package_name(feature_repo_url)
    return f"{feature_package_name}-{get_branch_or_commit(feature_repo_url)}"


# # TODO
# def get_model_name(model_config: dict):
#     """Generate a model name (str) 

#     Args:
#         model_config: the dict with model parameters

#     Returns:
#         str

#     """
#     model_list = []
#     for key, val in model_config.items():
#         if key == "type":
#             assert type(val) == str
#             # overall model type should always come first
#             model_list.insert(0, val)
#         if isinstance(val, dict):
#             if "type" in val:
#                 assert type(val["type"]) == str
#                 model_list.append(val["type"])

#     return "_".join(model_list)
