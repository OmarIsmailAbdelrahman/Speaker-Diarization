import os

BRANCH = 'main'
os.system('pip install wget')
os.system('apt-get -y install sox libsndfile1 ffmpeg')
os.system('pip install text-unidecode')
os.system('pip install "matplotlib>=3.3.2"')
os.system('pip install aiohttp==3.9.2')
os.system('pip install boto3 --upgrade')
os.system(f'python -m pip install git+https://github.com/motawie0/NeMo.git@{BRANCH}#egg=nemo_toolkit[all]')
os.system('pip install torchaudio -f https://download.pytorch.org/whl/torch_stable.html')
