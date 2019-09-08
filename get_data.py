from requests import Response, get
from re import compile, Pattern
import gzip
import os
import sys

"""
The only way to get a list of files from a directory via HTTP is by parsing the
raw HTML that the directory listing returns and scrounging the files from that.

I'm sorry.

You can pass arguments to the script to narrow down the files you want to
download. It will only get files that have names that contain at least one arg.
"""

URL = 'https://www.cs.york.ac.uk/aig/sw/gobnilp/data/'
DATA_DIR_NAME = 'data'

files: Response = get(URL)

if files.status_code == 200:
    regex: Pattern = compile('href="(.*\.gz)"')

    try:
        os.makedirs(DATA_DIR_NAME)
    except FileExistsError as error:
        print('Why do you ALREADY have a folder named gobnilp???')
        exit(1)

    for line in files.text.splitlines():
        results = regex.search(line)
        if results is not None and len(results.group()) > 1:
            filename = str(results.group(1))

            ignore = True
            
            for name in sys.argv[1:]:
                if name.lower() in filename.lower():
                    ignore = False
                    break

            if ignore:
                continue

            zipped_data_path = os.path.abspath(f'{DATA_DIR_NAME}/{filename}')
            data_path = os.path.abspath(f'{DATA_DIR_NAME}/{filename[:-3].lower()}')

            gz_file: Response = get(f'{URL}/{filename}')

            with open(zipped_data_path, 'wb') as zipped_file:
                zipped_file.write(gz_file.content)

            with gzip.open(zipped_data_path, 'rt') as zipped_file:
                with open(data_path, 'w') as data_file:
                    data_file.write(zipped_file.read())

            os.remove(zipped_data_path)
