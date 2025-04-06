import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile

from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    CacheConfig,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import FrameworkProcessor
from sagemaker.sklearn import SKLearn
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.model_monitor import ModelQualityMonitor, EndpointInput
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo, ConditionLessThan

from sagemaker.model_monitor import CronExpressionGenerator

from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables


import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Replace these values with your own
region = "us-east-2"
role = "arn:aws:iam::143176219551:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole"
default_bucket = "truck-eta-classification"
model_package_group_name = "TruckETAClassificationPackageGroup"
pipeline_name = "TruckETAClassification-StreamingDataPipeline"
base_job_prefix = "TruckETAClassification"
processing_instance_type = "ml.m5.xlarge"
training_instance_type = "ml.m5.xlarge"

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client

def get_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.describe_project(ProjectName=sagemaker_project_name)
        sagemaker_project_arn = response["ProjectArn"]
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags

def get_pipeline(
    region = "us-east-2",
    role = "arn:aws:iam::143176219551:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole",
    default_bucket = "truck-eta-classification",
    model_package_group_name = "TruckETAClassificationPackageGroup",
    pipeline_name = "TruckETAClassification-StreamingDataPipeline",
    base_job_prefix = "TruckETAClassification",
    processing_instance_type = "ml.m5.xlarge",
    training_instance_type = "ml.m5.xlarge",
    sagemaker_project_name=None,
):
    sagemaker_session = get_session(region, default_bucket)
    pipeline_session = get_pipeline_session(region, default_bucket)

    cache_config = CacheConfig(enable_caching=False, expire_after="PT1H")

    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
    config_file_data = ParameterString(name="ConfigFileURL", default_value="s3://truck-eta-classification/config/config.yaml")

    
    sklearn_processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="1.2-1",
        image_uri = "143176219551.dkr.ecr.us-east-2.amazonaws.com/custom-python310:latest",
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-truckETA",
        sagemaker_session=pipeline_session,
        role=role,
    )
    
    
    # Updating data for previous date 
    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="result1", source="/opt/ml/processing/result1"),
        ],
        code=os.path.join(BASE_DIR, "previous_data_updation.py"),
        arguments=["--config-file-data", config_file_data]
    )
    
    step_updation = ProcessingStep(
        name="PreviousDataUpdation",
        step_args=step_args,
    ) 
    
    
    # Fetch streaming data and dump to hopsworks
    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="result2", source="/opt/ml/processing/result2"),
        ],
        inputs=[
            ProcessingInput(
                source=step_updation.properties.ProcessingOutputConfig.Outputs["result1"].S3Output.S3Uri,
                destination="/opt/ml/processing/result1",
            ),
        ],
        code=os.path.join(BASE_DIR, "fetch_streaming_dump_to_hopsworks.py"),
        arguments=["--config-file-data", config_file_data]
    )

    step_fetch_streaming = ProcessingStep(
        name="FetchStreamingDumpToHopsworks",
        step_args=step_args,
    )  
    

    # Feature engineering and final merge store to hopsworks
    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="result3", source="/opt/ml/processing/result3"),
        ],
        inputs=[
            ProcessingInput(
                source=step_fetch_streaming.properties.ProcessingOutputConfig.Outputs["result2"].S3Output.S3Uri,
                destination="/opt/ml/processing/result2",
            ),
        ],
        code=os.path.join(BASE_DIR, "feature_engg_finalmerge_hopsworks.py"),
        arguments=["--config-file-data", config_file_data]
    )

    step_feature_engg = ProcessingStep(
        name="FeatureEnggFinalMergeHopsworks",
        step_args=step_args,
    )  

    
    
    # Checking First Day of the week
    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="evaluation1", source="/opt/ml/processing/evaluation1"),
        ],
        inputs=[
            ProcessingInput(
                source=step_feature_engg.properties.ProcessingOutputConfig.Outputs["result3"].S3Output.S3Uri,
                destination="/opt/ml/processing/result3",
            ),
        ],
        code=os.path.join(BASE_DIR, "is_first_day_of_week.py"),
        arguments=["--config-file-data", config_file_data]
    )
        
    evaluation_report1 = PropertyFile(
        name="IsFirstDayOfWeekReport",
        output_name="evaluation1",
        path="evaluation1.json",
    )

    step_check_first_day_week = ProcessingStep(
        name="IsFirstDayOfWeek",
        step_args=step_args,
        property_files=[evaluation_report1],

    )
    
    
    
    # NotFirstDay
    step_args = sklearn_processor.run(
        code=os.path.join(BASE_DIR, "not_first_day_of_week.py"),
        arguments=["--config-file-data", config_file_data]
    )
   
    step_not_first_day_week = ProcessingStep(
        name="NotFirstDayOfWeek",
        step_args=step_args,

    ) 
    
    
    # First day of week - Calculating Data Drift
    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="evaluation2", source="/opt/ml/processing/evaluation2"),
        ],
        code=os.path.join(BASE_DIR, "calculate_data_drift.py"),
        arguments=["--config-file-data", config_file_data]
    )
    
    evaluation_report2 = PropertyFile(
        name="IsDataDrifted",
        output_name="evaluation2",
        path="evaluation2.json",
    )
   
    step_calculate_data_drift = ProcessingStep(
        name="CalculateDataDrift",
        step_args=step_args,
        property_files=[evaluation_report2],
    )
    
    
    # condition step to check first day of the week or not
    cond_lte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_check_first_day_week.name,
            property_file=evaluation_report1,
            json_path="is_first_day_of_new_week"
        ),
        right=1,
    )
    step_condition_first_day = ConditionStep(
        name="CheckIsFirstDayOfWeek",
        conditions=[cond_lte],
        if_steps=[step_calculate_data_drift],
        else_steps=[step_not_first_day_week],
    )  
    
    # Calculating Model Drift
    step_args = sklearn_processor.run(
        outputs=[
            ProcessingOutput(output_name="evaluation3", source="/opt/ml/processing/evaluation3"),
        ],
        code=os.path.join(BASE_DIR, "calculate_model_drift.py"),
        arguments=["--config-file-data", config_file_data]
    )
    
    evaluation_report3 = PropertyFile(
        name="IsModelDrifted",
        output_name="evaluation3",
        path="evaluation3.json",
    )
   
    step_calculate_model_drift = ProcessingStep(
        name="CalculateModelDrift",
        step_args=step_args,
        property_files=[evaluation_report3],
    )
    
    
    # Not Model Drift
    step_args = sklearn_processor.run(
        code=os.path.join(BASE_DIR, "model_drift_not_detected.py"),
        arguments=["--config-file-data", config_file_data]
    )
    step_not_model_drift = ProcessingStep(
        name="ModelDriftNotDetected",
        step_args=step_args,
    )
    
    
    
    # Model retraining for Data Drift
    step_args = sklearn_processor.run(
        code=os.path.join(BASE_DIR, "model_retraining.py"),
        arguments=["--config-file-data", config_file_data]
    )
    
    step_model_retraining_data_drift = ProcessingStep(
        name="ModelRetrainingForDataDrift",
        step_args=step_args,
    )
  

    # Model retraining for Model drfit
    step_args = sklearn_processor.run(
        code=os.path.join(BASE_DIR, "model_retraining.py"),
        arguments=["--config-file-data", config_file_data]
    )
    
    step_model_retraining_model_drift = ProcessingStep(
        name="ModelRetrainingForModelDrift",
        step_args=step_args,
    )

    # condition step to check data drifted or not
    cond_lte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_calculate_data_drift.name,
            property_file=evaluation_report2,
            json_path="drifted"
        ),
        right=1,
    )
    step_condition_data_drifted = ConditionStep(
        name="CheckIfDataDrifted",
        conditions=[cond_lte],
        if_steps=[step_model_retraining_data_drift],
        else_steps=[step_calculate_model_drift],
    )
    
    
    
    # condition step to check model drifted or not
    cond_lte = ConditionLessThan(
        left=JsonGet(
            step_name=step_calculate_model_drift.name,
            property_file=evaluation_report3,
            json_path="f1_score"
        ),
        right=0.6,
    )
    step_condition_model_drifted = ConditionStep(
        name="CheckIfModelDrifted",
        conditions=[cond_lte],
        if_steps=[step_model_retraining_model_drift],
        else_steps=[step_not_model_drift],
    )
    


    # Pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            config_file_data,
        ],
        steps = [step_updation, step_fetch_streaming, step_feature_engg, step_check_first_day_week, step_condition_first_day, step_condition_data_drifted, step_condition_model_drifted],
        
        
        sagemaker_session=pipeline_session,
    )
    return pipeline
