Jetson Nano - First Time Setup
======
What you need:
- Jetson Nano
- SD card
- SD card adapter for computer
- USB keyboard
- monitor (HDMI)

Flashing Ubuntu Server disk image to the SD card
----
First we need to flash an SD card with a disk image. We're going to use Ubuntu Server 20.04. We can use [BalenaEtcher](https://www.balena.io/etcher/) to do this.
1. Download and install BalenaEtcher. 
2. Plug the SD card and adapter into your computer.
3. Open BalenaEtcher. Download and select the correct file
   - 4GB: https://developer.nvidia.com/jetson-nano-sd-card-image
   - 2GB: https://developer.nvidia.com/jetson-nano-2gb-sd-card-image
4. Select the device corresponding to the SD card.
5. Flash it. This will take ~20 minutes.
6. The SD card will automatically be unmounted after flashing completes. Remove the SD card and plug it in to the Jetson.

First boot and network configuration
----
Plug the Jetson in to power (USB-C for 2GB, Barrel Jack for 4GB) to boot. Plug in a USB keyboard and a monitor. You should eventually get a standard Ubuntu setup screen.

Jetson Nano requires using USB Wifi, but we need to do some configuration to connect to the CMU wireless network.
1. Find the device's mac address: `ifconfig wlan0 | grep ether`
2. Register the device on the CMU-DEVICES network [here](https://getonline.cmu.edu/hosts/register/wireless/).
3. Use `nmtui` to connect to CMU-DEVICES.
4. You should now be able to ssh into your device from your laptop: `ssh ubuntu@<hostname>.wifi.local.cmu.edu`

Install packages
----
- System update: `sudo apt upgrade`
- Some useful/necessary packages: `sudo apt -y install curl`
- Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Activate Rust: `source $HOME/.cargo/env`
- Update path by adding: `export PATH=$PATH:'/home/ubuntu/.local/bin'` to `.bashrc`

| Tool | `apt` | `pip3` | Test | 
| ---  | --- | --- | --- |
| Torch   | - | `wget https://nvidia.box.com/shared/static/h1z9sw4bb1ybi0rm3tu8qdj8hs05ljbm.whl -O torch-1.9.0-cp36-cp36m-linux_aarch64.whl`<br>`sudo apt-get install python3-pip libopenblas-base libopenmpi-dev`  | `pip3 install Cython`<br>`pip3 install numpy torch-1.9.0-cp36-cp36m-linux_aarch64.whl`|
| Transformers | - | `pip3 install setuptools-rust` <br> `pip3 install transformers` | `python3 generate_text.py`| 
| Camera       | `sudo apt install -y libgl1-mesa-glx` | `pip3 install opencv-python`<br> `sudo nano /boot/firmware/config.txt`, add line `start_x=1` and restart. | `python3 capture_photo.py`| 
| Torchvision | - |  | `python3 classify_image` |
| Microphone   | `sudo apt install -y libportaudio2` |  | `python3 capture_audio.py` |
| TTS      | `sudo apt install espeak`  | - |  `espeak "this is a test"`|
| SST      | `sudo apt install portaudio19-dev python3-pyaudio flac` | `pip3 install SpeechRecognition` | `python3 -m speech_recognition` |


Bugs
----
1. Speaker for espeak -- Can't switch outputs
2. Change camera resolution -- opencv dependencies issues?!

Set up CMU VPN for remote access
----
If the device stays on the CMU network but you need to access it from outside the CMU network, you will need to connect first to the CMU VPN. [Follow the instructions here](https://www.cmu.edu/computing/services/endpoint/network-access/vpn/how-to/).

Once you have the client installed and configured, you can connect and disconnect to the VPN from the command line using these aliases:
```
alias vpn_connect="printf '\n\nANDREW_PSWD\ny' | /opt/cisco/anyconnect/bin/vpn -s connect vpn.cmu.edu"
alias vpn_disconnect="/opt/cisco/anyconnect/bin/vpn disconnect"
```


