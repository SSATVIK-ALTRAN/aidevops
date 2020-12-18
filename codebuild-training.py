import boto3
import re

import os
#import numpy as np
#import pandas as pd

from sagemaker import get_execution_role
import sagemaker as sage


role = get_execution_role()
sess = sage.Session()

account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name
image = '{}.dkr.ecr.{}.amazonaws.com/cop-group9/sagemaker-demo'.format(account, region)

clf = sage.estimator.Estimator(image,
                               role, 1, 'ml.c4.2xlarge',
                               output_path="s3://cop-group9/model",
                               sagemaker_session=sess)

#clf.fit("s3://sagemaker-aidevops/training-data")
clf.fit({'training': 's3://cop-group9/training-data'})

print('Finished training job: ', clf._current_job_name)




