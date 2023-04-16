import os
import subprocess
import concurrent.futures
import logging
import argparse

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to load")
    parser.add_argument('--temperature', type=float, default=1.0, help="Start temperature")
    parser.add_argument('--seq_len', type=int, default=256, help="The length of extracted sequence")
    parser.add_argument('--top_k', type=int, default=40, help="sample from the top_k tokens output by the model")

    return parser.parse_args()

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    return (output, error)

def save_output(output, file_path):
    with open(file_path, "w") as f:
        f.write(output)

if __name__ == '__main__':
    root_dir = './'
    args = parse_arguments()
    config = {
        'model': args.model,
        'temp': args.temperature,
        'len': args.seq_len,
        'k': args.top_k
    }

    generated_folder = '{}-temp{}-len{}-k{}'.format(config['model'], config['temp'], config['len'], config['k'])
    
    data_dir = os.path.join(root_dir, 'clone/save/codeparrot/codeparrot-clean')
    # path to the training data
    tool_path = os.path.join(root_dir, 'clone/simian-2.5.10.jar')
    # path to the clone detection tool



    step = 2 # * 100000
    start = 0
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
        # the number of training data split is 53
        for i in range(53):
            data_path = os.path.join(data_dir, '{}'.format(i))
            assert os.path.exists(data_path), "File {} does not exist".format(data_path)
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
