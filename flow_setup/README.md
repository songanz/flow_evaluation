# Install docker follow until step 4
https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04 

## Setup docker
In your home directory, copy and paste the `flow_setup`

Then builds the docker
```bash
docker build --squash --compress -t flow:1.0 . 
docker images
```
## Setup docker home
```bash
mkdir docker_home_flow
cd docker_home_flow
vi .bashrc
```
Copy paste the following contents inside the new bashrc file you created for the flow docker
```bash
export PYTHONPATH=$SUMO_TOOLS:/usr/local/src/flow
export PATH=$SUMO_BIN:$PATH
conda activate flow
```
Then close the vi 
## start the docker 
```bash
cd ~/flow_setup/scripts
./docker_vnc_na.sh -p 44 -d ~/docker_home_flow/ -i flow:1.0 -g 1920x1080
```
If you get an output stating `YOU ARE NOW RUNNING ON DOCKER` then its good

## Setup the vncviewer
Download and install deb x64 version of vncviewer (for Ubuntu) there is a dropdown list click on it and download deb x64 version
### Launch vncviewer 
New connection enter the path <your computer name>:5944 you should be able access docker
Inside docker open terminal it should already show (flow) check sumo, sumo-gui, etc.. 
```bash
cd $FLOW_HOME
jupyter notebook
```
You can go through all the flow tutorials here
