## Running Experiments


Note: with deep speed configuration, we find out a newer version of accelerate works better. 
Install the required dependencies:
   ```
   pip install accelerate==0.31.0
   ```

#### DeepSpeed Configuration

We use DeepSpeed for efficient distributed training. You can find an example configuration in `./ds_config.yaml`. This configuration helps optimize memory usage and computation speed across multiple GPUs and nodes.

### Single Node with Multiple GPUs Job

To run a job on a single node with multiple GPUs:

```bash
sbatch submit-deepspeed-singlenode.sh
```

This Slurm job will:
1. Allocate a single node
2. Activate the training environment
3. Run `accelerate launch train_stage_1.py --config configs/train/stage1.yaml`

**Note:** We have tested this configuration on 4 GPU instances (e.g., g5.24xlarge). For these instances, adjust `train_width: 768 train_height: 768` and set `use_8bit_adam: False` in your configuration file.

### Multi-Node with Multiple GPUs Job

To run a job across multiple nodes, each with multiple GPUs:

```bash
sbatch submit-deepspeed-multinode.sh
```

**Note:** We have tested this distribution with two g5.24xlarge instances.

