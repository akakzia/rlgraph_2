import os
from time import sleep

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)


job_directory = "%s/.job" % os.getcwd()
scratch = os.environ['SCRATCH']

# Make top level directories
mkdir_p(job_directory)

nb_seeds = 5
models = ['flat', 'full_gn', 'relation_network', 'deep_sets', 'interaction_network_2']

for i in range(nb_seeds):
    for model in models:
        job_file = os.path.join(job_directory, "continuous_{}%.slurm".format(model))

        with open(job_file, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH --account=kcr@gpu\n")
            fh.writelines("#SBATCH --job-name=continuous_{}\n".format(model))
            fh.writelines("#SBATCH --qos=qos_gpu-t3\n")
            fh.writelines("#SBATCH --output=continuous_{}%_%j.out\n".format(model))
            fh.writelines("#SBATCH --error=continuous_{}%_%j.out\n".format(model))
            fh.writelines("#SBATCH --time=19:59:59\n")
            fh.writelines("#SBATCH --ntasks=24\n")
            fh.writelines("#SBATCH --ntasks-per-node=1\n")
            fh.writelines("#SBATCH --gres=gpu:1\n")
            fh.writelines("#SBATCH --hint=nomultithread\n")
            fh.writelines("#SBATCH --array=0-0\n")

            fh.writelines("module load pytorch-gpu/py3/1.4.0\n")

            fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
            fh.writelines("export LIBRARY_PATH=$LIBRARY_PATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/lib\n")
            fh.writelines("export CPATH=$CPATH:/gpfslocalsup/spack_soft/mesa/18.3.6/gcc-9.1.0-bikg6w3g2be2otzrmyy43zddre4jahme/include\n")
            fh.writelines("export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/linkhome/rech/genisi01/uqy56ga/.mujoco/mujoco200/bin\n")
            fh.writelines("export OMPI_MCA_opal_warn_on_missing_libcuda=0\n")
            fh.writelines("export OMPI_MCA_btl_openib_allow_ib=1\n")
            fh.writelines("export OMPI_MCA_btl_openib_warn_default_gid_prefix=0\n")
            fh.writelines("export OMPI_MCA_mpi_warn_on_fork=0\n")

            fh.writelines("srun python -u -B train.py --algo 'continuous' --n-blocks 5 --n-epochs 1000 --n-cycles 50 --n-batches 30 --architecture {} --save-dir 'continuous_{}/' 2>&1 ".format(model, model))

        os.system("sbatch %s" % job_file)
        sleep(1)
        
