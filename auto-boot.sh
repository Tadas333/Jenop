#!/bin/bash

#echo $PATH

cd ~


#echo 'jenoptik' | sudo -S mount -t cifs -o username=Manager,password=Jenoptik33478 //192.168.1.223/recordings /home/mari1/S_001/deep_storage

#cd /home/mari1/UTMC_Sender/ && sudo ./UTMC\ Sender && sudo python3 /home/mari1/S_001/switch_002.py &> /home/mari1/S_001/log.txt

#cd /home/mari1/UTMC_Sender

#sudo gnome-terminal cd UTMC_Sender

#sudo gnome-terminal -e /home/mari1/UTMC_Sender/
#UTMC\ Sender

echo 'jenoptik' | sudo -S python3 /home/mari1/S_001/switch_002.py &> /home/mari1/S_001/log.txt
