import subprocess

if __name__ == '__main__':
    cmd = 'horovodrun --check-build'
    subprocess.run(cmd, shell=True)