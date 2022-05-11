import argparse
import utils
from torchvision.datasets.utils import download_and_extract_archive

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='facades', help='Name of the Pix2Pix dataset')
    args = parser.parse_args()

    # set dataset downloading and extracting location
    dataset_name = args.dataset_name
    url = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'
    filename = f'dataset_name.tar.gz'
    download_root = utils.images_path

    # download and extract data
    download_and_extract_archive(url = url, download_root = download_root, filename = filename, remove_finished = True)