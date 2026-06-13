import sys
import os

from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.constants import HF_TOKEN, HF_REPO_ID

from huggingface_hub import HfApi


class ModelPusher:
    def __init__(self,
                 model_pusher_config: ModelPusherConfig,
                 model_evaluation_artifact: ModelEvaluationArtifact):
        """
        :param model_pusher_config: Configuration for model pusher
        :param model_evaluation_artifact: Output reference of model evaluation artifact stage
        """
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
            self.hf_api = HfApi()
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Method Name :   initiate_model_pusher
        Description :   This function initiates model pusher steps —
                        uploads the trained model to HuggingFace Hub.

        Output      :   Returns ModelPusherArtifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Entered initiate_model_pusher method of ModelPusher class")
            print("----------------------------------------------------------")
            print("Starting Model Pusher Component")

            trained_model_path = self.model_evaluation_artifact.trained_model_path

            # Ensure the model file exists
            if not os.path.exists(trained_model_path):
                raise Exception(f"Trained model file not found at: {trained_model_path}")

            # Ensure HuggingFace repo exists, create if not
            try:
                self.hf_api.repo_info(repo_id=HF_REPO_ID, repo_type="model", token=HF_TOKEN)
                logging.info(f"HuggingFace repo '{HF_REPO_ID}' already exists.")
            except Exception:
                logging.info(f"Repo '{HF_REPO_ID}' not found. Creating new repo on HuggingFace Hub.")
                self.hf_api.create_repo(
                    repo_id=HF_REPO_ID,
                    token=HF_TOKEN,
                    repo_type="model",
                    private=True
                )
                logging.info(f"Repo '{HF_REPO_ID}' created successfully.")

            # Upload model to HuggingFace Hub
            logging.info(f"Uploading model to HuggingFace Hub: {HF_REPO_ID}")
            self.hf_api.upload_file(
                path_or_fileobj=trained_model_path,
                path_in_repo="model.pkl",
                repo_id=HF_REPO_ID,
                token=HF_TOKEN,
                repo_type="model"
            )
            logging.info("Model uploaded successfully to HuggingFace Hub.")

            # Also save model locally to model pusher dir
            import shutil
            os.makedirs(self.model_pusher_config.model_pusher_dir, exist_ok=True)
            local_model_path = os.path.join(self.model_pusher_config.model_pusher_dir, "model.pkl")
            shutil.copy(trained_model_path, local_model_path)
            logging.info(f"Model also saved locally at: {local_model_path}")

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=HF_REPO_ID,
                s3_model_path="model.pkl"
            )

            logging.info(f"Model pusher artifact: {model_pusher_artifact}")
            logging.info("Exited initiate_model_pusher method of ModelPusher class")
            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys) from e