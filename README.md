# RuuningIPNonHPC

Here is the script changes I made to the testing scripts and training scripts  and lossfunc.py for changing/trying out different loss fucntions  and printing confusion matrix.

So the files you could look are:

train.py

test_faz.py

test_rv.py

lossfunc.py

my .sh files to see what configurations on HPC i use like nodes, cpu, time, gpu, (turns out using -n 1 and  -c 10 is a good balance where the model runs quite fast) 

Though it  is upto anyone's preference, since Omnia uses -n 10, so one can experiment and see whatever seems faster since during peak hours using large configuration will delay ur job on queue in HPC. 

Also  you can add ur email on  .sh script so you get notified when the file ends running to  make it easier for yourself.








