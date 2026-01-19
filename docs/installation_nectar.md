# Installing AF3 on Nectar

Ensure you have an instance with a reasonably large GPU reservation, I recommend at least 40 GB vram. 

### Set up working directory

Perform all installation and work in ephemeral /mnt directory or volume, home directory is too small. Ensure permissions are correct and download repo.

```sh
cd /mnt
sudo chmod -R 777 .
git clone https://github.com/Pav-mis/alphafold3_nectar
```

Get data and parameters:

```sh
mkdir -p af3_data
#transfer genetic databases or run:
/mnt/alphafold3_nectar/fetch_databases.sh af3_data
#I would recommend transfering from aws bucket to pawsey, then scp from pawsey to nectar, as I had trouble installing aws-cli on nectar


mkdir -p models
#transfer parameter file af3.bin.zst to models dir
```

### Check that GPU is visible to instance

Sometimes this doesn't work, check or you will waste time. If you get an error, reset your instance.

```sh
nvidia-smi -q
```

### Install Docker

```sh
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

Add the repository to apt sources:

```sh
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo docker run hello-world
```

### Enable Rootless Docker

```sh
sudo apt-get install -y uidmap systemd-container

sudo machinectl shell $(whoami)@ /bin/bash -c 'dockerd-rootless-setuptool.sh install && sudo loginctl enable-linger $(whoami) && DOCKER_HOST=unix:///run/user/1001/docker.sock docker context use rootless'
```

### Install NVIDIA Drivers

```sh
sudo apt-get -y install alsa-utils ubuntu-drivers-common
sudo ubuntu-drivers install

sudo nvidia-smi --gpu-reset

nvidia-smi  # Check that the drivers are installed.
```

Accept the "Pending kernel upgrade" dialog if it appears.

You will need to reboot the instance with `sudo reboot now` to reset the GPU if
you see the following warning:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
Make sure that the latest NVIDIA driver is installed and running.
```

Proceed only if `nvidia-smi` has a sensible output.

### Installing NVIDIA Support for Docker

```sh
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
systemctl --user restart docker
sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
```

Check that your container can see the GPU:

```sh
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

this should produce similar looking output to running just `nvidia-smi`.

### Creating AF3 docker container

This part requires some intervention, as docker will try to build in the home directory. Do the following.

Stop docker

```sh
systemctl --user stop docker.service
```

Create new data directory

```sh
mkdir -p /mnt/docker-root
```

Edit your daemon configuration file to point to new directory.

```sh
nano ~/.config/docker/daemon.json
```

Update it to look like this:

```json
{
    "data-root": "/mnt/docker-root",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
```

Restart Docker

```sh
systemctl --user daemon-reload
systemctl --user start docker.service
```

Check that the new location is being used 

```sh
docker info | grep "Docker Root Dir"
```

Build AF3 container:

```sh
docker build -t alphafold3 -f docker/Dockerfile .
```

### Check it works

Run the following to make sure AF3 works:

```sh
docker run -it \
    --volume /mnt/alphafold3_nectar/test/af_input:/root/af_input \
    --volume /mnt/alphafold3_nectar/test/af_output:/root/af_output \
    --volume /mnt/models:/root/models \
    --volume /mnt/af3_data/af3_data:/root/public_databases \
    --gpus all \
    alphafold3 \
    python run_alphafold.py \
    --json_path=/root/af_input/test.json \
    --model_dir=/root/models \
    --output_dir=/root/af_output
```

Alternatively, run the test script in `./test/scripts` which will also run GPU utilization monitoring. You should expect to see a spike in GPU usage towards the end of the task, as the initial stages of AF3 inference are CPU-dependant.

```sh
#ensure pip, pandas, and matplotlib are installed
sudo apt install python3-pip
pip install pandas
pip install matplotlib

#run the script
bash /mnt/alphafold3_nectar/test/scripts/test_af3.sh
```

You may need to update the script to modify csv/plot output and AF3 run parameters.






