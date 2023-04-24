import os
import subprocess
import concurrent.futures
import logging
import argparse
import re
import hashlib

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to load")
    parser.add_argument('--temperature', type=float, default=1.0, help="Start temperature")
    parser.add_argument('--seq_len', type=int, default=256, help="The length of extracted sequence")
    parser.add_argument('--top_k', type=int, default=40, help="sample from the top_k tokens output by the model")

    parser.add_argument('--internet-sampling', action='store_true', help="condition the generation on the internet")
    parser.add_argument('--prompt_mode', type=str, default="single_md5",choices=["single_md5","direct_prompt"], help="The mode of the prompt to use for generation")
    parser.add_argument('--prompt', type=str, default="", help="The prompt to use for generation(can also be the path to a file containing the prompt)")
    parser.add_argument('--prompt_hash', type=str, default="", help="The hash of the prompt to use for generation")
    parser.add_argument("--tool_path",type=str,default=None)

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
    # if not args.internet_sampling:
    #     generated_folder = '{}-temp{}-len{}-k{}'.format(config['model'], config['temp'], config['len'], config['k'])
    # else:
    #     if args.prompt_mode == 'single_md5':
    #         hash_value = args.prompt_hash
    #     elif args.prompt_mode == 'direct_prompt':
    #         hash_value = hashlib.sha1(args.prompt.encode('utf-8')).hexdigest()
    #         args.prompt_hash = hash_value
    #     generated_folder = '{}-temp{}-len{}-k{}/internet/{}'.format(config['model'], config['temp'], config['len'], config['k'], hash_value)
    
    # path to the training data & the clone detection tool
    if args.tool_path is None:
        tool_path = os.path.join(root_dir, 'clone/simian-2.5.10.jar')
        data_dir = os.path.join(root_dir, 'clone/save/codeparrot/codeparrot-clean')
    else:
        tool_path = args.tool_path
        data_dir = tool_path.replace("simian-2.5.10.jar","save/codeparrot/codeparrot-clean")

    if args.internet_sampling:
        if args.prompt_mode == 'direct_prompt':
            if isinstance(args.prompt, str) and len(args.prompt) == 40 and re.match("^[a-f0-9]+$", args.prompt):
                args.prompt_hash = args.prompt
            else:
                hash_value = hashlib.sha1(args.prompt.encode('utf-8')).hexdigest()
                args.prompt_hash = hash_value




    step = 2 # * 100000
    start = 0
    end = 20 # * 100000
    # process in chunks
    for size in range(start, end, step):
        # _start = size * 100000
        # _end = (size + step) * 100000
        _start = size * 100000 + 1
        _end = (size + step) * 100000 + 96
        if not args.internet_sampling:
            file_path = os.path.join(root_dir, 'extract/results/{}/all_{}-{}'.format(generated_folder, _start, _end))
        else:
            file_path = os.path.join(root_dir, 'extract/results/{}/all_{}-{}-{}'.format(generated_folder,args.prompt_hash,_start, _end))
            if not os.path.exists(file_path):
                file_path = os.path.join(root_dir, 'results/{}/all_{}-{}-{}'.format(generated_folder,args.prompt_hash,_start, _end))
        print(os.path.abspath(file_path))
        assert os.path.exists(file_path), "File {} does not exist".format(file_path)

        # create log dir
        if not args.internet_sampling:
            log_dir = os.path.join(root_dir, 'log/save/', generated_folder, "{}-{}".format(_start, _end))
        else:
            log_dir = os.path.join(root_dir, 'log/save/', generated_folder,args.prompt_hash, "{}-{}".format(_start, _end))
        logger.info(f'[save dict]: {log_dir}')
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
