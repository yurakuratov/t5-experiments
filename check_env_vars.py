import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    var = 'MY_ENV_VARIABLE'
    print(f"{var}: {os.environ.get(var, 'is not set')}")