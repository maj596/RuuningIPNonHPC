# RuuningIPNonHPC

Here is the script changes I made to the testing scripts and training scripts  and lossfunc.py for changing/trying out different loss fucntions  and printing confusion matrix.

So the files you could look are:

train.py

test_faz.py

test_rv.py

lossfunc.py

my .sh files to see what configurations on HPC i use like nodes, cpu, time, gpu, (turns out using -n 1 and  -c 10 is a good balance where the model runs quite fast) 

Though it  is upto anyone's preference, since Omnia uses -n 10, so one can experiment and see whatever seems faster since during peak hours using large configuration will delay ur job on queue in HPC. 

This configuration always works even during peak hours when a lot of users are running jobs since its a good amount of resources to use.

#SBATCH -n 1

#SBATCH -c 10

#SBATCH -t 70:00:00

#SBATCH -p nvidia

#SBATCH --gres=gpu:v100:1

#SBATCH --mem=50G

#SBATCH --output=slurm_%j.out

#SBATCH --error=slurm_%j.err

#SBATCH --mail-type=FAIL,END

#SBATCH --mail-user=maj596@nyu.edu

For IPN V1 the gpu does not matter u can use either V100 or A100, but in IPNV2 its a mess, since IPNV2 gives different errors/warnings for different GPU and different pytorch versions and different CUDA versions, since we dont know the version authors used, we are forced to assume and I have yet to find a configuration that doesnt have some error wether its giving some random index error 5 hours into training or training metrics just staying as 0. My notion page showcases some problems I faced with different CUDA and pytorch version. I hit a wall and decided to focus on just IPN-1 since none of the CUDA or pytorch version I tried ended up working correctly. But maybe u will approach this problem differently and figure out the correct pytorch and other modules to use.

Also you can add ur email on  .sh script so you get notified when the file ends running to  make it easier for yourself.

Also I didnt upload the log files since those are too large to upload here, but you can easily run and test the model using the google doc guide Omnia made.

ipnv1 guide: https://docs.google.com/document/d/1dmheYCaPSNuEwJKtdrjEW_HmsZEsCBmQCj7u65_hmPQ/edit (tesing works with author's checkpoint, training is buggy) (better to stick  with this)

ipnv2 guide: https://docs.google.com/document/u/2/d/1nPxpW4d4uHlv2Z1okkWgJK8GqOb4vb4EHFNqu5OPM0U/edit  (no authors checkpoints, training is buggy (so can't do testing), picking module versions is up to guesswork)








