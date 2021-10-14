import subprocess

if __name__ == '__main__':
    cmd = 'pip list'
    subprocess.run(cmd, shell=True)