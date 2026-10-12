from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from fmf_project.engine import run_training

if __name__=="__main__":run_training("finetune")
