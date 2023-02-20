import os
import subprocess
import concurrent.futures
import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return (output, error)

def save_output(output, file_path):
    with open(file_path, "w") as f:
        f.write(output)

if __name__ == '__main__':
    root_dir = './'
    config = {'model': 'codeparrot/codeparrot-small', 'temp': 1.0, 'len': 512, 'k':22}

    generated_folder = '{}-temp{}-len{}-k{}'.format(config['model'], config['temp'], config['len'], config['k'])
    
    data_dir = os.path.join(root_dir, 'clone/save/codeparrot/codeparrot-clean')
    tool_path = os.path.join(root_dir, 'clone/simian-2.5.10.jar')



    step = 2 # * 100000
    start = 4
    end = 20 # * 100000
    # process in chunks
    for size in range(start, end, step):
        _start = size * 100000
        _end = (size + step) * 100000
        file_path = os.path.join(root_dir, 'extract/results/{}/all_{}-{}'.format(generated_folder, _start, _end))
        assert os.path.exists(file_path), "File {} does not exist".format(file_path)

        # create log dir
        log_dir = os.path.join(root_dir, 'log/save/', generated_folder, "{}-{}".format(_start, _end))
        os.makedirs(log_dir, exist_ok=True)

        # build the commands to run
        commands = []
        for data in os.listdir(data_dir):
            data_path = os.path.join(data_dir, data)
            commands.append(['java', '-jar', tool_path, data_path, file_path])

        logger.info("Batch {}-{} started".format(_start, _end))
        
        # launch the commands in parallel using a process pool
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_command, command) for command in commands]

        # wait for all futures to complete and capture the output
        outputs = [future.result() for future in futures]

        # decode the output and error into strings (assuming utf-8 encoding)
        output_strs = [output.decode("utf-8") for output, error in outputs]
        error_strs = [error.decode("utf-8") for output, error in outputs]

        # save output to files in parallel
        logger.info("Saving output to {}".format(log_dir))
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(save_output, output_str, 
                os.path.join(log_dir, "{}.log".format(i))) for i, output_str in enumerate(output_strs)]
            
        # free memory
        del outputs
        del output_strs
        del error_strs


        logger.info("Batch {}-{} finished".format(_start, _end))
