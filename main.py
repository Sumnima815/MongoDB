import sys
from src.exception import MyException
from src.logger import logging
from src.pipeline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    try:
        logging.info("Starting training pipeline")
        pipeline = TrainPipeline()
        pipeline.run_pipeline()
        logging.info("Pipeline completed successfully")
    except Exception as e:
        raise MyException(e, sys)