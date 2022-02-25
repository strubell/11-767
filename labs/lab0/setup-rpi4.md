Raspberry Pi - First Time Setup
======
What you need:
- Raspberry Pi
- SD card
- SD card adapter for computer
- USB keyboard
- monitor (HDMI)
- micro HDMI to HDMI-A cable

Flashing Ubuntu Server disk image to the SD card
----
First we need to flash an SD card with a disk image. We're going to use Ubuntu Server 20.04. We can use [BalenaEtcher](https://www.balena.io/etcher/) to do this.
1. Download and install BalenaEtcher. 
2. Plug the SD card and adapter into your computer.
3. Open BalenaEtcher. Select *Flash from URL* then enter the URL: `http://herron.lti.cs.cmu.edu/~strubell/11-767/ubuntu_server_20.04.2_wifi.zip`
4. Select the device corresponding to the SD card.
5. Flash it. This will take a few minutes.
6. The SD card will automatically be unmounted after flashing completes. Remove the SD card and plug it in to the Pi.

First boot and network configuration
----
Plug the Pi in to power (USB-C) to boot. Plug in a USB keyboard and a monitor. You should eventually get a login prompt (if there's a bunch of other text on the screen, hit enter to get the login prompt).

Username is ubuntu. Password is on the board.

Raspberry Pi 4 comes with on-board wifi, but we need to do some configuration to connect to the CMU wireless network.
1. Find the device's mac address: `ifconfig wlan0 | grep ether`
2. Register the device on the CMU-DEVICES network [here](https://getonline.cmu.edu/hosts/register/wireless/).
3. Use `nmtui` to connect to CMU-DEVICES.
4. You should now be able to ssh into your device from your laptop: `ssh ubuntu@<hostname>.wifi.local.cmu.edu`

Install packages
----
- System update: `sudo apt upgrade`
- Some useful/necessary packages: `sudo apt -y install gcc g++ python3-pip python3-rpi.gpio`
- Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Activate Rust: `source $HOME/.cargo/env`
- Update path by adding: `export PATH=$PATH:'/home/ubuntu/.local/bin'` to `.bashrc`

| Tool | `apt` | `pip3` | Test | 
| ---  | --- | --- | --- |
| Torch 1.10   | - | `pip3 install https://tiger.lti.cs.cmu.edu/ybisk/11-767/torch-1.10.0a0+git38c1851-cp38-cp38-linux_aarch64.whl` | |
| Transformers | - | `pip3 install transformers` | `python3 generate_text.py`| 
| Camera       | `sudo apt install -y libgl1-mesa-glx` | `pip3 install opencv-python`<br> `sudo nano /boot/firmware/config.txt`, add line `start_x=1` and restart. | `python3 capture_photo.py`| 
| Torchvision | - | `pip3 install torchvision` | `python3 classify_image` |
| Microphone   | `sudo apt install -y libportaudio2` | `pip3 install sounddevice scipy` | `python3 capture_audio.py` |
| TTS      | `sudo apt install espeak`  | - |  `espeak "this is a test"`|
| SST      | `sudo apt install portaudio19-dev python3-pyaudio flac` | `pip3 install SpeechRecognition` | `python3 -m speech_recognition` |


**I2C**
```sh
$ sudo apt-get install -y i2c-tools
$ sudo i2cdetect -F 1
Functionalities implemented by /dev/i2c-1:
I2C                              yes
SMBus Quick Command              yes
SMBus Send Byte                  yes
SMBus Receive Byte               yes
SMBus Write Byte                 yes
SMBus Read Byte                  yes
SMBus Write Word                 yes
SMBus Read Word                  yes
SMBus Process Call               yes
SMBus Block Write                yes
SMBus Block Read                 no
SMBus Block Process Call         no
SMBus PEC                        yes
I2C Block Write                  yes
I2C Block Read                   yes
$ sudo pip3 install adafruit-circuitpython-motorkit
$ wget https://raw.githubusercontent.com/adafruit/Adafruit_CircuitPython_MotorKit/master/examples/motorkit_robot.py
$ wget https://raw.githubusercontent.com/adafruit/Adafruit_CircuitPython_MotorKit/master/examples/motorkit_robot_test.py
$ sudo python3 motorkit_robot_test.py
```

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


