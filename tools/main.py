from dotenv import load_dotenv

load_dotenv()  # reads variables from a .env file and sets them in os.environ
# ruff: noqa: E402
import os.path as osp
import sys

sys.path.append(osp.dirname(osp.dirname(__file__)))
from tools.runner import Runner


def main(argv=None):
    Runner(argv=argv).run()


if __name__ == "__main__":
    main()
