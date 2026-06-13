import sys
import os
import dill
from typing import Optional
from sklearn.metrics import f1_score

from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import (DataIngestionArtifact, ModelEvaluationArtifact,
                                         ModelTrainerArtifact, ClassificationMetricArtifact)
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.main_utils import load_object
from src.constants import TARGET_COLUMN, MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE, HF_TOKEN, HF_REPO_ID

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


class ModelEvaluation:
    def __init__(self,
                 model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys)

    def get_best_model_from_huggingface(self) -> Optional[object]:
        """
        Fetches the best/production model from HuggingFace Hub if it exists.
        Returns None if no model exists yet.
        """
        try:
            logging.info("Checking for existing model on HuggingFace Hub")

            try:
                repo_files = list(list_repo_files(
                    repo_id=HF_REPO_ID,
                    token=HF_TOKEN,
                    repo_type="model"
                ))
            except Exception:
                logging.info("HuggingFace repo not found or empty. Treating as no existing model.")
                return None

            model_file = "model.pkl"
            if model_file not in repo_files:
                logging.info("No existing model found on HuggingFace Hub.")
                return None

            logging.info("Downloading existing model from HuggingFace Hub")
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=model_file,
                token=HF_TOKEN,
                repo_type="model"
            )

            with open(downloaded_path, "rb") as f:
                model = dill.load(f)
            logging.info("Existing model loaded successfully from HuggingFace Hub")
            return model

        except Exception as e:
            logging.info(f"Could not fetch model from HuggingFace: {e}")
            return None

    def _preprocess_test_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply same preprocessing steps as data_transformation.py"""

        # Map gender
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)

        # Create dummy variables (do NOT drop id column)
        df = pd.get_dummies(df, drop_first=True)

        # Rename columns
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })

        # Cast dummy columns to int
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')

        # Reorder columns to match exact training order
        expected_columns = [
            'id', 'Gender', 'Age', 'Driving_License', 'Region_Code',
            'Previously_Insured', 'Annual_Premium', 'Policy_Sales_Channel',
            'Vintage', 'Vehicle_Age_lt_1_Year', 'Vehicle_Age_gt_2_Years',
            'Vehicle_Damage_Yes'
        ]
        df = df[expected_columns]

        return df

    def evaluate_model(self) -> ModelEvaluationArtifact:
        """
        Compares newly trained model against the existing production model.
        If new model performs better by threshold, it is accepted.
        """
        try:
            logging.info("Loading test data for evaluation")

            # Load and preprocess test data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x_test = test_df.drop(columns=[TARGET_COLUMN])
            y_test = test_df[TARGET_COLUMN]
            logging.info("Test data loaded and split into features and target")

            x_test = self._preprocess_test_data(x_test)
            logging.info("Preprocessing applied to test data")

            # Evaluate newly trained model
            logging.info("Loading newly trained model")
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1 = f1_score(y_test, trained_model.predict(x_test))
            logging.info(f"Newly trained model F1 Score: {trained_model_f1}")

            # Fetch and evaluate existing production model from HuggingFace
            best_model_f1 = 0.0
            best_model = self.get_best_model_from_huggingface()

            if best_model is not None:
                best_model_f1 = f1_score(y_test, best_model.predict(x_test))
                logging.info(f"Existing production model F1 Score: {best_model_f1}")
            else:
                logging.info("No existing production model found. New model will be accepted by default.")

            # Compare models
            improved_accuracy = trained_model_f1 - best_model_f1
            is_model_accepted = improved_accuracy >= MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE

            logging.info(f"Model improvement: {improved_accuracy:.4f} (threshold: {MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE})")
            logging.info(f"Is new model accepted: {is_model_accepted}")

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                changed_accuracy=improved_accuracy,
                best_model_path=self.model_trainer_artifact.trained_model_file_path if is_model_accepted else None,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                train_model_metric_artifact=self.model_trainer_artifact.metric_artifact,
                best_model_metric_artifact=ClassificationMetricArtifact(
                    f1_score=best_model_f1,
                    precision_score=0.0,
                    recall_score=0.0
                )
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        This function initiates the model evaluation steps.
        """
        try:
            logging.info("Entered initiate_model_evaluation method of ModelEvaluation class")
            print("----------------------------------------------------------")
            print("Starting Model Evaluation Component")

            model_evaluation_artifact = self.evaluate_model()

            logging.info("Model evaluation completed successfully")
            logging.info("Exited initiate_model_evaluation method of ModelEvaluation class")
            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys) from e