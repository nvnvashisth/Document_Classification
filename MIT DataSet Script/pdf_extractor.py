import glob
import os
import pandas as pd
from urllib.parse import urlparse
from io import StringIO, BytesIO
from zipfile import ZipFile
from urllib.request import urlopen


cur_dir = os.getcwd()

# Read the csv
df = pd.read_csv('data.csv')

for course_link in df['download_url']:
    print("Downloading and processing",course_link,"...")
    course_zip = urlopen(course_link)
    with ZipFile(BytesIO(course_zip.read())) as zip_file:
        file_names = zip_file.namelist()
        for file_name in file_names:
            if file_name.endswith('pdf'):
                if ('lecture' in file_name) or ('reading' in file_name):

                    #Split path name to get course_id and the file name
                    path = urlparse(file_name).path.strip('/').split('/')
                    
                    #Create directory with course ID
                    if not os.path.exists(cur_dir+'/'+path[0]):
                        os.makedirs(cur_dir+'/'+path[0])

                    # Write file to the folder with name as course_id
                    content = zip_file.open(file_name).read()
                    open(cur_dir+'/'+path[0]+'/'+path[-1], 'wb').write(content)
