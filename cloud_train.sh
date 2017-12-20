#! /bin/bash

DATE=`date +"%Y%m%d%H%M%S"`
FILENAME="cnn_cifar10_$DATE"
export BUCKET_NAME=keras-ml-models
export JOB_NAME=$FILENAME
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-central1

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --region $REGION \
    --runtime-version 1.4 \
    --module-name trainer.main \
    --config=trainer/config.yaml \
    --package-path ./trainer \
    -- \
    --train-file gs://$BUCKET_NAME/cifar10-data
