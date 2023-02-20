import os
import subprocess
import concurrent.futures

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return (output, error)

if __name__ == '__main__':
    root_dir = './'
    config = {'model': 'codeparrot/codeparrot-small', 'temp': 1.0, 'len': 512, 'k':21}

    generated_folder = '{}-temp{}-len{}-k{}'.format(config['model'], config['temp'], config['len'], config['k'])
    
    data_dir = os.path.join(root_dir, 'clone/save/codeparrot/codeparrot-clean')
    tool_path = os.path.join(root_dir, 'clone/simian-2.5.10.jar')

    size = 10000
    log_dir = os.path.join(root_dir, 'log/save/', generated_folder, str(size))
    os.makedirs(log_dir, exist_ok=True)

    # build the commands to run
    commands = []
    for data in os.listdir(data_dir):
        data_path = os.path.join(data_dir, data)
        log_path = os.path.join(log_dir, data+".log")
        generated_path = os.path.join(root_dir, 'extract/results/{}/all_{}'.format(generated_folder, size))
        assert os.path.exists(generated_path)
        commands.append(['java', '-jar', tool_path, data_path, generated_path])
        # commands.append(['java', '-jar', tool_path, '/mnt/hdd1/zyang/Privacy-in-Code-Models/extract/utils.py'])
        print(commands[-1])
    
    # launch the commands in parallel using a process pool
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_command, command) for command in commands]

    # wait for all futures to complete and capture the output
    outputs = [future.result() for future in futures]

    # decode the output and error into strings (assuming utf-8 encoding)
    output_strs = [output.decode("utf-8") for output, error in outputs]
    error_strs = [error.decode("utf-8") for output, error in outputs]

    # save output to files
    for i, output_str in enumerate(output_strs):
        with open(os.path.join(log_dir, "{}.log".format(i)), "w") as f:
            f.write(output_str)


    print("Java command finished")
