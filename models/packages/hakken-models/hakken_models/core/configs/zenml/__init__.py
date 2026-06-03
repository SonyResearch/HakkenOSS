from enum import StrEnum

from kubernetes.client.models import V1Toleration
from zenml.config import DockerSettings
from zenml.integrations.kubernetes.flavors.kubernetes_orchestrator_flavor import (
    KubernetesOrchestratorSettings,
)
from zenml.integrations.mlflow.flavors.mlflow_experiment_tracker_flavor import (
    MLFlowExperimentTrackerSettings,
)


class KubernetesKind(StrEnum):
    GPU = "gpu"
    IN_CLUSTER = "in_cluster"


class OrchestratorSettings:
    @staticmethod
    def kubernetes(kind: KubernetesKind = KubernetesKind.GPU) -> KubernetesOrchestratorSettings:
        if kind == KubernetesKind.GPU:
            return KubernetesOrchestratorSettings(
                pod_settings={
                    "node_selectors": {
                        "sai-gpu": "true",
                    },
                    "tolerations": [
                        V1Toleration(
                            key="sai-gpu", operator="Equal", value="true", effect="NoExecute"
                        ),
                    ],
                }
            )
        if kind == KubernetesKind.IN_CLUSTER:
            return KubernetesOrchestratorSettings(
                pod_settings={
                    "env": [
                        {"name": "ZENML_STORE_URL", "value": "http://mvp-zenml:8080"},
                        {"name": "ZENML_STORE_TYPE", "value": "rest"},
                        {"name": "MLFLOW_TRACKING_URI", "value": "http://mvp-mlflow:5000"},
                    ]
                },
                orchestrator_pod_settings={
                    "env": [
                        {"name": "ZENML_STORE_URL", "value": "http://mvp-zenml:8080"},
                        {"name": "ZENML_STORE_TYPE", "value": "rest"},
                        {"name": "MLFLOW_TRACKING_URI", "value": "http://mvp-mlflow:5000"},
                    ]
                },
            )

        raise ValueError(f"Unsupported Kubernetes kind: {kind}")


class ContainerSettings:
    @staticmethod
    def docker() -> DockerSettings:
        return DockerSettings(
            dockerfile="./Dockerfile.zenml",
            build_context_root="./",
            parent_image_build_config={
                "build_options": {
                    "secret": [
                        "id=codeartifact_username,env=UV_INDEX_HAKKEN_PIP_REGISTRY_USERNAME",
                        "id=codeartifact_password,env=UV_INDEX_HAKKEN_PIP_REGISTRY_PASSWORD",
                    ]
                },
            },
            image_tag="hakken-models",
            skip_build=False,  # Ensure ZenML doesn't skip the build step
            force_build=True,  # Forces a new build even if settings look the same
            prevent_build_reuse=True,  # Prevents reusing previous builds
        )


class ExperimentTrackerSettings:
    @staticmethod
    def mlflow(experiment_name: str = "kge") -> MLFlowExperimentTrackerSettings:
        return MLFlowExperimentTrackerSettings(experiment_name=experiment_name, nested=False)
