from pathlib import Path

import pandas as pd
import pytest
from kedro.framework.context import KedroContext
from kedro.framework.hooks import _create_hook_manager
from kedro.runner import SequentialRunner


@pytest.fixture
def project_context(config_loader):
    return KedroContext(
        package_name="sentiment_analysis_twitter",
        project_path=Path.cwd(),
        config_loader=config_loader,
        hook_manager=_create_hook_manager(),
    )


@pytest.fixture
def conf_catalog(project_context):
    return project_context.catalog


@pytest.fixture
def conf_params(project_context):
    return project_context.params


@pytest.fixture
def seq_runner():
    return SequentialRunner()


@pytest.fixture
def tweets():
    return pd.DataFrame(
        {
            "text": [
                "I love this movie!",
                "I hate this movie!",
                "I love this movie!",
                "I hate this movie!",
            ],
            "sentiment": [1, 0, 1, 0],
        }
    )


@pytest.fixture
def cleaned_tweets():
    return pd.DataFrame(
        {
            "text": [
                "love movie",
                "hate movie",
                "love movie",
                "love movie",
            ],
            "sentiment": [1, 0, 1, 0],
        }
    )


@pytest.fixture
def X_train():
    return pd.DataFrame(
        {
            "text": [
                "love movie",
                "hate movie",
                "love movie",
            ],
        }
    )


@pytest.fixture
def X_test():
    return pd.DataFrame(
        {
            "text": [
                "love movie",
            ],
        }
    )
