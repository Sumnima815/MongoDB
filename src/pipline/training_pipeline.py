import sys
from src.exception import MyException
from src.logger import logging

from src.components.data_ingestion import DataIngestion

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

        # self.data_validation_config = DataValidationConfig()
        # self.data_transformation_config = DataTransformationConfig()
        # self.model_trainer_config = ModelTrainerConfig()
        # self.model_evaluation_config = ModelEvaluationConfig()
        # self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting
        the data ingestion component.
        """
        try:
            logging.info(
                "Entered the start_data_ingestion method of TrainPipeline class"
            )

            logging.info("Getting the data from MongoDB")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )

            data_ingestion_artifact = (
                data_ingestion.initiate_data_ingestion()
            )

            logging.info("Got the train_set and test_set from MongoDB")
            logging.info(
                "Exited the start_data_ingestion method of TrainPipeline class"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def run_pipeline(self) -> None:
        """
        This method of TrainPipeline class is responsible for running
        the complete pipeline.
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            # data_validation_artifact = self.start_data_validation(
            #     data_ingestion_artifact=data_ingestion_artifact
            # )

            # data_transformation_artifact = self.start_data_transformation(
            #     data_ingestion_artifact=data_ingestion_artifact,
            #     data_validation_artifact=data_validation_artifact
            # )

            # model_trainer_artifact = self.start_model_trainer(
            #     data_transformation_artifact=data_transformation_artifact
            # )

            # model_evaluation_artifact = self.start_model_evaluation(
            #     data_ingestion_artifact=data_ingestion_artifact,
            #     model_trainer_artifact=model_trainer_artifact
            # )

            # if not model_evaluation_artifact.is_model_accepted:
            #     logging.info("Model not accepted.")
            #     return

            # model_pusher_artifact = self.start_model_pusher(
            #     model_evaluation_artifact=model_evaluation_artifact
            # )

            logging.info("Pipeline executed successfully.")

        except Exception as e:
            raise MyException(e, sys) from e