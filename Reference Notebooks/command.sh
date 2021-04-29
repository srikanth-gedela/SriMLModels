# demo-04-DeployingPipelineModelWithFlask

server.py

test_data.json

{
    "x" : ["I hate Brokeback Mountain"]
}

curl -XPOST http://localhost:8080/api -H 'Content-Type: application/json' -d @test_data.json

{
    "x" : ["I hate Brokeback Mountain", "I love Brokeback Mountain"]
}

curl -XPOST http://localhost:8080/api -H 'Content-Type: application/json' -d @test_data.json

request.py

# demo-06-SettingUpCloudStorageBucketAndUploadingSavedModel

cloud.user@loonycorn.com
loony-classifier-models
models
decision_tree_model
linear_svc_model
logistic_regression_model

# demo-07-DeployingAnSklearnModelToAServerlessEnvironmentOnGCP

sentimental_analysis

{
	"model" : ["DecisionClassifier"],
	"x" : ["I love Brokeback Mountain"]
}

test_data_with_model.json

curl -XPOST https://us-central1-deploying-ml-solutions.cloudfunctions.net/sentimental_analysis  -H 'Content-Type: application/json' -d @test_data_with_model.json

LinearSVC

curl -XPOST https://us-central1-deploying-ml-solutions.cloudfunctions.net/sentimental_analysis  -H 'Content-Type: application/json' -d @test_data_with_model.json

"I dislike Brokeback Mountain"

curl -XPOST https://us-central1-deploying-ml-solutions.cloudfunctions.net/sentimental_analysis  -H 'Content-Type: application/json' -d @test_data_with_model.json

curl -XPOST https://us-central1-deploying-ml-solutions.cloudfunctions.net/sentimental_analysis  -H 'Content-Type: application/json' -d @test_data_with_model.json

# demo-08-EnablingApiAndUploadingEvaluationData

evaluation_data
Cloud Machine Learning Engine
Data Labeling API

# demo-09-CreatingModelAndVersion

sentiment_analysis_model
Sentiment analysis using scikit-learn
v1
Using decision trees for sentiment analysis
dataset
evaluation_job
deploying-ml-solutions.dataset.evaluation_table

# demo-10-SamplingModelPrediction

{
    "instances" : ["These Harry Potter movies really suck"]
}

{
    "instances" : ["These Harry Potter movies really suck", 
                   "I love Brokeback Mountain"]
}

# demo-11-DeployingModelAndVersion

export PS1="\[\e[34m\]\w\[\e[m\]>\n-->"

MODEL_NAME="sentiment_analysis_model"

VERSION_NAME="v1"

gcloud ai-platform versions describe $VERSION_NAME \
  --model $MODEL_NAME

nano input.json

"These Harry Potter movies really suck"

INPUT_DATA_FILE="input.json"

gcloud ai-platform predict \
--model $MODEL_NAME \
--version $VERSION_NAME \
--json-instances $INPUT_DATA_FILE

input.json

"I love Brokeback Mountain"

gcloud ai-platform predict \
--model $MODEL_NAME \
--version $VERSION_NAME \
--json-instances $INPUT_DATA_FILE

# demo-12-DeployingAnotherVersionOfModel

MODEL_DIR="gs://loony-classifier-models/models/linear_svc_model/"
VERSION_NAME="v2"
MODEL_NAME="sentiment_analysis_model"
FRAMEWORK="scikit-learn"

gcloud ai-platform versions create $VERSION_NAME \
  --model $MODEL_NAME \
  --origin $MODEL_DIR \
  --runtime-version=1.13 \
  --framework $FRAMEWORK \
  --python-version=3.5

gcloud ai-platform versions describe $VERSION_NAME \
  --model $MODEL_NAME

curl -X POST -H "Content-Type: application/json" \
-d '{"instances" : ["These Harry Potter movies really suck", "I love Brokeback Mountain"]}' \
-H "Authorization: Bearer $(gcloud auth application-default print-access-token)" \
https://ml.googleapis.com/v1/projects/deploying-ml-solutions/models/sentiment_analysis_model/versions/v2:predict

# demo-13-GettingOnlinePredictionsViaREST

access_token=$(gcloud auth application-default print-access-token)

echo $access_token

curl -X POST \
-d '{"instances" : ["These Harry Potter movies really suck", "I love Brokeback Mountain"]}' \
https://ml.googleapis.com/v1/projects/deploying-ml-solutions/models/sentiment_analysis_model/versions/v1:predict\?access_token=ya29.GqUBeAeQGGgLAkMnpW1GCDWC3xso-LWv_rYFTDehs6WW54Y_6LP7cJOqplEIQoRWzyNX2iHBy8_fdDLh6eXVdyqOU-n1GNauGRpMNiZA4alOfWxYbpOgRRp7NYJC-r3Y2DVQyQVsCVk3Q3zr3RtWPJRQbZGCG8y69InDFRow2Df_Q1LsjImDkDGq-Lkn5LDvF-R2FMmsYDXZOjx8bj7PHtVRp-xfGcq0

# demo-14-MonitoringADeployedModelUsingStackdriverAndIdentifyingCommonPerformanceMonitoringMetrics

sentiment_analysis_metrics

# demo-15-CreatingAnAmazonSageMakerNotebookInstance

deployment-instance

# demo-16-BuildingADeepLearningModelUsingTensorflowAndDeployingToHostingService

mnist.py

# demo-17-MonitoringPerformanceUsingAWSCloudTrail

mnist-model-trail
mnist-model-trail-data

