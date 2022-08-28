
# Install code server similar to VS Code
○ curl -fsSL https://code-server.dev/install.sh | sh
To have systemd start code-server now and restart on boot:
   sudo systemctl enable --now code-server@$USER
Or, if you don't want/need a background service you can run: code-server

# Create Virtual environment
conda create -n sriMLModels python=3.9

# install jupyter lab & Python Kernel
conda install -c conda-forge jupyterlab
python -m ipykernel install --user --name sriMLModels

# Add all required libraries for the module to this file and execute. You can add versions to make it more strict 
pip install -r requirements.txt
