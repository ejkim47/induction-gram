from os.path import join, expanduser, dirname
import os

if 'ALM_DIR' in os.environ:
    ALM_DIR = os.environ['ALM_DIR']
else:
    ALM_DIR = join(dirname(dirname(os.path.abspath(__file__))))

DATA_DIR_ROOT = join(ALM_DIR, 'data')
SAVE_DIR_ROOT = join(ALM_DIR, 'results')

# individual datasets...
BABYLM_DATA_DIR = join(DATA_DIR_ROOT, 'babylm')

DATA_DIR_ROOT = join(ALM_DIR, 'Data')
INFINIGRAM_INDEX_PATH = join(ALM_DIR, 'infini-gram-indexes')
SAVE_DIR_ROOT = join(ALM_DIR, 'Exp')

# hugging face token
TOKEN_HF = ""
