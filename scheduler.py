import itertools
import collections
import glob
import os
import datetime
import subprocess
import string
import sys


def get_memory(combo):
    if combo['dataset'] == 'clothing':
        return 48000
    else:
        return 30000


def get_cpu(combo):
    if combo['dataset'] == 'clothing':
        return 5
    else:
        return 3


def get_config(combo):
    if combo['dataset'] == 'clothing':
        return 'configs/clothing.yaml'
    elif combo['dataset'] == 'cifar':
        return 'configs/cifar10.yaml'
    elif combo['dataset'] == 'cifar100':
        return 'configs/cifar100.yaml'

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def get_run_id():
    filename = "logs/expts.txt"
    if os.path.isfile(filename) is False:
        with open(filename, 'w') as f:
            f.write("")
        return 0
    else:
        with open(filename, 'r') as f:
            expts = f.readlines()
        run_id = len(expts)
    return run_id

hyperparameters = [
    [('problem',), ['mwnet','finetune']],
    [('encoder_type',), ['imagenet', 'simclr']],
    [('loss',), ['qloss','ce']],
    [('dataset',), ["cifar","cifar100"]],#
    [('corruption_prob',), [0,0.1, 0.2,0.3, 0.4, 0.5, 0.8, 0.9, 0.95]],#[]],
    [('q',), [ 0.5, 0.66]],
    [('corruption_type',), ['unif']],
    [('seed',), [220,221,222,223,224]],#
]

def get_gpu(combo):
    if combo['problem'] =='mwnet':
        return "2080ti"
    if combo['problem'] =='finetune':
        return "titanx"
    return "1080ti"
    
def is_valid(combo):
    if combo['corruption_type'] in {'flip', 'asym'} and combo['corruption_prob'] not in {0.2, 0.3, 0.4}:
        return False
    if combo['corruption_type']=='flip2' and combo['corruption_prob'] not in {0.2, 0.3, 0.4, 0.6}:
        return False
    if combo['corruption_type']=='unif' and combo['corruption_prob'] not in {0, 0.2, 0.5, 0.8, 0.9, 0.95}:
        return False
    if combo['corruption_type']=='unif' and combo['corruption_prob'] ==0:
        if combo['problem']=='mwnet' or combo['loss'] =='qloss':
            return False
    if combo['problem'] =='finetune':
        if combo['loss'] =='ce':
            if combo['q'] ==0.66:
                return True
            return False
        else:
            return True
    if combo['problem'] =='mwnet':
        if combo['loss'] =='ce' and combo['q'] ==0.66:
            return True
        return False
    



other_dependencies = {'gpu': get_gpu, 'memory': get_memory, 'n_cpu':get_cpu, 'config':get_config, 'valid':is_valid}

run_id = int(get_run_id())

key_hyperparameters = [x[0] for x in hyperparameters]
value_hyperparameters = [x[1] for x in hyperparameters]
combinations = list(itertools.product(*value_hyperparameters))

scripts = []
gpu_counts =collections.defaultdict(int)

for combo in combinations:
    # Write the scheduler scripts
    with open("scripts/template.sh", 'r') as f:
        schedule_script = f.read()

    combo = {k[0]: v for (k, v) in zip(key_hyperparameters, combo)}

    for k, v in other_dependencies.items():
         combo[k] = v(combo)
    if not combo['valid']:
        #print(combo)
        continue
    combo['run_id'] = run_id
    gpu_counts[combo['gpu']] +=1

    for k, v in combo.items():
        if "{%s}" % k in schedule_script:
            schedule_script = schedule_script.replace("{%s}" % k, str(v))
    


    schedule_script += "\n"

    # Write schedule script
    script_name = 'slurm/cv_%d.sh' % run_id
    with open(script_name, 'w') as f:
        f.write(schedule_script)
    scripts.append(script_name)

    # Making files executable
    subprocess.check_output('chmod +x %s' % script_name, shell=True)

    # Update experiment logs
    output = "Script Name = " + script_name +", Time Now= "+ datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n" 
    with open("logs/expts.txt", "a") as f:
        f.write(output)
    # For the next job
    run_id += 1

print(gpu_counts)
# schedule jobs
for script in scripts:
    command = "sbatch %s" % script
    #print(command)
    print(subprocess.check_output(command, shell=True))
