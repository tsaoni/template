import requests
import gdown

# link: https://drive.google.com/file/d/10_FDsWmWdeWZI-AavTbW54y2CLpKBUCz/view?usp=share_link
# check file size ls -lh <file_name>
file_id = "1URNq8vGbhDNBhu_UfD9HrEK8bkgWcqpM" # replace with your file id
url = "https://drive.google.com/uc?id=" + file_id
output_file = './dataset/politeness.csv'


def read_file(output_file):
    with open(output_file, 'r', encoding='utf-8', errors='replace') as file:
        lines = file.readlines()
        import pdb 
        pdb.set_trace()

def use_gdown(url, output_file):
    gdown.download(url, output_file, quiet=False)

def use_request():
    response = requests.get(url)
    if response.status_code == 200:
        with open("your_file_name.extension", "wb") as f:
            f.write(response.content)
            print("Download successful!")
    else:
        print("Download failed!")

use_gdown(url, output_file)
read_file(output_file)