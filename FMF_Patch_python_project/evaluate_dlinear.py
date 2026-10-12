from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from dlinear_project.engine import evaluate_entry
if __name__=='__main__':evaluate_entry()
