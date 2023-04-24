import sys
import json

sys.path.append("./hifigan")
from inference import load_checkpoint
from models import Generator
from env import AttrDict

# Load Generator
checkpoint_dict = load_checkpoint("./g_02500000", 'cpu')
with open('./hifigan/config_v1.json') as f:
    json_config = json.loads(f.read())
h = AttrDict(json_config)
generator = Generator(h)
generator.load_state_dict(checkpoint_dict['generator'])
