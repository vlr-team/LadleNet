#! /bin/bash
source $PROJECT/miniconda3/etc/profile.d/conda.sh
conda activate vlr
accelerate launch /ocean/projects/cis220039p/ayanovic/vlr_project/LadleNet/Model/LadleNet+/LadleNet+.py