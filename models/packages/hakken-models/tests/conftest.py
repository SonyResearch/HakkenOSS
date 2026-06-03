import os


def pytest_configure():
    os.environ["ZENML_DISABLE_STEP_LOGS_STORAGE"] = "true"
    os.environ["ZENML_DISABLE_PIPELINE_LOGS_STORAGE"] = "true"
