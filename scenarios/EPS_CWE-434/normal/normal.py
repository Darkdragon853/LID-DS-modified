import argparse
import glob
import random
import subprocess
import sys
import time
import os

parser = argparse.ArgumentParser(description='HTTPS-Client Simulation.')

parser.add_argument('-ip', dest='server_ip', action='store', type=str, required=True,
                    help='The IP address of the target server')

args = parser.parse_args()


path_download_images = "/home/images/"
target_port = "8000"
target_url = args.server_ip


def do_normal():
    """ sends random eps files from path_download_images to the victim """
    print("Sending random Image")
    file_list = glob.glob(path_download_images + "/*.eps")
    choice = random.choice(file_list)
    send_file(choice, "http://" + target_url + ":" + target_port)
    print("Sent: " + choice)


def send_file(file, victim):
    """ sends the given file to the victim"""
    print("sending: " + file + " to " + victim)
    os.system("curl -X PUT --upload-file " + file + " " + victim)


while True:
    try:
        sys.stdin.readline()
        do_normal()
    except Exception as e:
        print(e)
        time.sleep(1)
