# keras-cnn-cifar10
A Keras-TensorFlow Convolutional Neural Network used for training and testing on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

CIFAR10 dataset includes:

* 50,000 32x32 color training images.
* 10,000 32x32 color testing images.
* 10 different classes.
* 6,000 images per class.

## Getting Started
The below instructions are required for everyone, they will get you installed and running locally. Additionally, 
if you plan to train in the Google Cloud ML Engine (GCMLE) environment, please see the section titled 
**Google Cloud ML Engine Deployment**.

## Prerequisites
#### Mac
For Mac installations, it's recommended you use homebrew to install HDF5 first.
```
brew tap homebrew/science
brew install hdf5
```

## Installation
```
virtualenv keras-cifar10
. ./keras-cifar10/bin/activate
pip install --upgrade pip setuptools
git clone https://github.com/drewszurko/keras-cnn-cifar10.git
cd keras-cnn-cifar10/ 
pip install -r requirements.txt
python setup.py install
python trainer/main.py
```

If you plan to train locally on an NVIDIA® GPU, you'll need to `pip install --upgrade tensorflow-gpu`.

Note: Executing both `python install -r requirements.txt` and `python setup.py install` is necessary. TensorFlow can't be included in the
`setup.py` -> `install_requires` because of an install conflict with the GCMLE environment. 


## Google Cloud ML Engine Deployment
If you plan to train on the GCMLE environment, there's a few additional steps you'll need to take.

### Prerequisites
Required for GCMLE deployment:
* A Google Cloud Platform account access.
* Your GCMLE and Cloud Storage APIs have been activated. 
* Cloud SDK has been installed and initialized locally.
* A Cloud Storage Bucket for our CIFAR10 data.

If you are unsure that you meet the prerequisites above, read the 
[Getting Started](https://cloud.google.com/ml-engine/docs/getting-started-training-prediction) guide from Google, specifically:

* Before you begin (ignore the 'TensorFlow installed' part).
* Set up your Cloud Storage bucket.
* Costs.

After you've completed the above steps, you'll need to download and untar the
[CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) dataset file. 

### Deployment
After this section, you should be up and running on the GCMLE environment.

#### Prepare data
First, create a subfolder in your GC Storage Bucket. This subfolder will hold your train/test data.

Copy the extracted CIFAR10 dataset files and upload them to the previous steps folder:
```
- batches.meta
- data_batch_1
- data_batch_2
- data_batch_3
- data_batch_4
- data_batch_5
- test_batch
```

#### Modify files
Replace the below ##[] with your GCMLE project info. 

**setup.py**
```
setup(name='trainer',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      include_package_data=True,
      description='##[PROJECT_DESCRIPTION]',
      author='##[YOUR_NAME]', 
      zip_safe=False)
```

**cloud_train.sh**
```
DATE=`date +"%Y%m%d%H%M%S"`
FILENAME="##[MODEL_NAME]_$DATE"
export BUCKET_NAME=##[STORAGE_BUCKET_NAME] 
export JOB_NAME=$FILENAME
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=##[SERVER_REGIONS] (https://cloud.google.com/ml-engine/docs/regions. Note: GPUs not suppored in all regions.)

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --region $REGION \
    --runtime-version 1.4 \
    --module-name trainer.main \
    --config=trainer/config.yaml \
    --package-path ./trainer \
    -- \
    --train-file gs://$BUCKET_NAME/##[STORAGE_BUCKET_SUBFOLDER]
```

**load_data.py**

`_DATA_DIR_CLOUD = 'gs://##[STORAGE_BUCKET]/##[STORAGE_BUCKET_SUBFOLDER]'`

**main.py**

`cloud_train = True`
    
#### Run in the GCMLE
After you have completed the steps above, run `./cloud_train.sh` from your terminal.
 If successful, you should see a similar prompt:

```
Job [JOB NAME] submitted successfully.
Your job is still active. You may view the status of your job with the command

  $ gcloud ml-engine jobs describe [JOB_NAME]

or continue streaming the logs with the command

  $ gcloud ml-engine jobs stream-logs [JOB_NAME]
jobId: [JOB_NAME]
state: QUEUED
```

Note: The default polling interval for `$ gcloud ml-engine jobs stream-logs [YOUR_JOB_NAME]` is 60 seconds. 
If you want to poll more frequently, you can use  `$ gcloud ml-engine jobs stream-logs [YOUR_JOB_NAME] --polling-interval = [INSERT SECONDS HERE]`.

## Known Issues
* `Tensorflow` module cannot be included in the `setup.py` -> `install_requires` because of a conflict with the GCMLE environment. 
The `Tensorflow` module will override your TensorFlow version request `cloud_train.sh` -> `--runtime-version 1.4 \`. This 
results in the version of TensorFlow that only supports CPU training, and it's extremely slowwwwwwww.

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/drewszurko/keras-cnn-cifar10/blob/master/LICENSE) file for details.

## Acknowledgments
Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton - The creators of the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.