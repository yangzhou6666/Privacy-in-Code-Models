import os


if __name__ == '__main__':
    root_dir = './'
    data_dir = os.path.join(root_dir, 'clone/save/codeparrot/codeparrot-clean')
    tool_path = os.path.join(root_dir, 'clone/simian-2.5.10.jar')
    log_dir = os.path.join(root_dir, 'log/save/codeparrot/codeparrot-clean')
    os.makedirs(log_dir, exist_ok=True)
    for data in os.listdir(data_dir):
        data_path = os.path.join(data_dir, data)
        log_path = os.path.join(log_dir, data+".log")
        generated_path = os.path.join(root_dir, 'extract/results/codeparrot/codeparrot-small-temp1.0-len256-k40/all')
        cmd = 'java -jar {} {} {} > {} &'.format(tool_path, data_path, generated_path ,log_path)
        print(cmd)
        os.system(cmd)