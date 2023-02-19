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

    commands = []
    for data in os.listdir(data_dir):
        data_path = os.path.join(data_dir, data)
        log_path = os.path.join(log_dir, data+".log")
        generated_path = os.path.join(root_dir, 'extract/results/{}/all_{}'.format(generated_folder, size))
        assert os.path.exists(generated_path)
        commands.append(['java', '-jar', tool_path, data_path, generated_path, '>', os.path.join(log_dir, data+".log")])
        print(commands[-1])
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_command, command) for command in commands]


    outputs = [future.result() for future in futures]

    print("Java command finished")
