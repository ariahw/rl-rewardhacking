# Convenience script to setup a GPU environment

source .env.gpu

# Run installation of basic packages
apt-get update
apt-get install -y vim
apt-get install -y git
apt-get install -y tmux
apt-get install -y unzip

# Load other environment variables
pip install uv
uv venv $VENV_DIR
source $VENV_DIR/bin/activate
source setup.sh