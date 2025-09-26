import os
import sys
import zipfile


if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve


def download_tinyImg200(path,
                        url='http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                        tarname='tiny-imagenet-200.zip'):
    if not os.path.exists(path):
        os.mkdir(path)

    output_name = os.path.join(path, tarname)
    if os.path.exists(output_name):
        print("Dataset was already downloaded to '{}'. Skip downloading".format(output_name))
    else:
        urlretrieve(url, output_name)
        print("Dataset was downloaded to '{}'".format(output_name))

    print("Extract downloaded dataset to '{}'".format(path))
    with zipfile.ZipFile(output_name, 'r') as f:
        f.extractall(path=path)
