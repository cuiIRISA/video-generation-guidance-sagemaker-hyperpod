# video-generation-guidance-sagemaker-hyperpod



### SageMaker Hyperpod Cluster Creation Guide

This guide explains how to create and manage a SageMaker Hyperpod cluster based on the [SageMaker Hyperpod workshop studio guidance](https://catalog.workshops.aws/sagemaker-hyperpod/en-US).

## Cluster Creation


### Lifecycle Scripts

Lifecycle scripts provide flexibility to customize your cluster during creation. You can use them to:
- Install software packages
- Set up necessary configurations
- Configure Slurm
- Create users
- Install Conda or Docker

Download the lifecycle scripts and upload to S3 bucket
   
    git clone --depth=1 https://github.com/aws-samples/awsome-distributed-training/
    cd awsome-distributed-training/1.architectures/5.sagemaker-hyperpod/LifecycleScripts/
    aws s3 cp --recursive base-config/ s3://${BUCKET}/src
    
### Cluster configuration Scripts
    
1. Prepare the required configuration in a `cluster-config.json` file and `provisioning_parameters.json`, adjusting requirements as needed.

2. copy to the S3 Bucket 
   ```
   aws s3 cp provisioning_parameters.json s3://${BUCKET}/src/
   ```
3. Create the cluster using AWS CLI:
   ```
   aws sagemaker create-cluster --cli-input-json file://cluster-config.json --region $AWS_REGION
   ```


See one example of `cluster-config.json` file and `provisioning_parameters.json` at here. 

### Scaling the Cluster

To increase the number of worker instances:

1. Update the `cluster-config.json` file with the new instance count.

2. Run the update command:
   ```
    aws sagemaker update-cluster --cluster-name $my-cluster-name --instance-groups file://update-cluster-config.json --region $AWS_REGION
   ```

### Shutting Down the Cluster

Once your production training workload is finished, shut down the entire cluster:

```
aws sagemaker delete-cluster --cluster-name $my-cluster-name
```


### Notes


SageMaker HyperPod supports the attachment of a shared file system, such as Amazon FSx for Lustre. This integration brings several benefits to your machine learning workflow. FSx for Lustre enables [full bi-directional synchronization with Amazon S3](https://aws.amazon.com/blogs/aws/enhanced-amazon-s3-integration-for-amazon-fsx-for-lustre/), including the synchronization of deleted files and objects. It also allows you to synchronize file systems with multiple S3 buckets or prefixes, providing a unified view across multiple datasets. 

- Ensure you have the necessary AWS CLI permissions and configurations set up.
- Always review and test your cluster configuration before deploying to production. 
- Monitor your cluster usage to optimize costs and performance.

For more detailed information and advanced configurations, refer to the [SageMaker Hyperpod workshop studio](https://catalog.workshops.aws/sagemaker-hyperpod/en-US).

## Cluster login


Following the guidance and setup on [Access your SageMaker HyperPod cluster nodes](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-hyperpod-run-jobs-access-nodes.html) 

### SSH into controller node 

You can log into the Hyperpod cluster controller node by running 

./easy-ssh.sh -c controller-machine ml-cluster
sudo su - ubuntu

You can setup the VS Code connection directly with SSH. [In this guide](https://catalog.us-east-1.prod.workshops.aws/workshops/e3752eec-63b5-4033-9720-fa68d35164e9/en-US/05-advanced/05-vs-code) you can setup a SSH Proxy via SSM and use that to connect in Visual Studio Code.

### SSH into worker node 

If it is the first time that you login the controller node, you need to generate a new keypair and then copy it to authorized_keys file 

#### on headnode
   ```
    cd ~/.ssh
    ssh-keygen -t rsa -q -f "$HOME/.ssh/id_rsa" -N ""
    cat id_rsa.pub >> authorized_keys
   ```

salloc -N 1
ssh $(srun hostname)

#### on worker node
 
Install Mini Conda 

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh -b -f -p ~/miniconda3

source ~/miniconda3/bin/activate
conda create -n videogen python=3.10

conda activate videogen


### Slurm command

List existing partitions and nodes per partition. 
sinfo

List jobs in the queues or running. 
squeue



## Running the AnimateAnyone

Following the implementation https://github.com/MooreThreads/Moore-AnimateAnyone, you can install all the required packages, download the pre-trained weights, and test the training script. 

If the training script worked as expected, you can then run the scheduled job with sbatch, we provide a batch file to simulate the single GPU running job.  


## Setup

1. Activate the conda environment 

    source ~/miniconda3/bin/activate
    conda activate videogen

    
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

2. Download pre-trained weights:
   ```
   python tools/download_weights.py
   ```

3. Test the training script:
   ```
   accelerate launch train_stage_1.py --config configs/train/stage1.yaml
   accelerate launch train_stage_2.py --config configs/train/stage2.yaml
   ```

## Running Experiments

### Single Node Job (with single or multiple GPUs)

To run a single Node job, use the provided batch file:

```
sbatch submit-animateanyone-algo.sh
```

When using the single GPUs with small GPU memory instance e.g. G5 2xlarge, you might need to set a smaller 
train_bs: 2 and train_width: 128 train_height: 128. Otherwise you might encounter Out of Memory issue. 


### Hyperparameter Testing

For hyperparameter testing, use:

```
sbatch submit-hyperparameter-testing.sh
```

## Monitoring Experiments

To monitor and visualize your experiments, use MLflow:

```
mlflow ui --backend-store-uri ./mlruns/
```

This will launch the MLflow UI, allowing you to track and compare your experiment runs.

## Additional Resources

- For detailed installation instructions, data preparation, and inference guides, refer to the [original repository](https://github.com/MooreThreads/Moore-AnimateAnyone).
- Make sure to check the repository for any updates or changes to the implementation.

## Notes

- Ensure you have the necessary GPU resources and CUDA setup before running the experiments.
- Adjust the batch files and configurations as needed for your specific environment and requirements.
