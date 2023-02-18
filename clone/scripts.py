import os


if __name__ == '__main__':
    root_dir = './'
    config = {'model': 'codeparrot/codeparrot', 'temp': 1.0, 'len': 1024, 'k':40}
    generated_folder = '{}-temp{}-len{}-k{}'.format(config['model'], config['temp'], config['len'], config['k'])
    
    data_dir = os.path.join(root_dir, 'clone/save/codeparrot/codeparrot-clean')
    tool_path = os.path.join(root_dir, 'clone/simian-2.5.10.jar')
    log_dir = os.path.join(root_dir, 'log/save/', generated_folder)

    
    os.makedirs(log_dir, exist_ok=True)
    for data in os.listdir(data_dir):
        data_path = os.path.join(data_dir, data)
        log_path = os.path.join(log_dir, data+".log")
        generated_path = os.path.join(root_dir, 'extract/results/{}/all'.format(generated_folder))
        cmd = 'java -jar {} {} {} > {} &'.format(tool_path, data_path, generated_path ,log_path)
        print(cmd)
        os.system(cmd)