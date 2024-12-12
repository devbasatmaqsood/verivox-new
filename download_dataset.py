"""
AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os, gdown

if __name__ == "__main__":
    # cmd = "curl -o ./LA.zip -# https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip\?sequence\=3\&isAllowed\=y"
    # os.system(cmd)
    # cmd = "unzip LA.zip"
    # os.system(cmd)
    
    # url_train = ""
    # url_val = "https://drive.google.com/drive/u/0/folders/1-QC5N7sBWTmJsa9poXtlZPJFp5bZ1Fo6"
    # url_test = "https://drive.google.com/drive/u/0/folders/1-JWdjBaHJTB5i5WskBBlSCJT0EHDIlX3"
    
    # output_train = ""
    # output_val = "IDASVspoofing_val_joined.zip"
    # output_test = "IDASVspoofing_test_joined.zip"
    
    # gdown.download(url_train, output_train, quiet=False)
    # gdown.download(url_val, output_val, quiet=False, fuzzy=True)
    # gdown.download(url_test, output_test, quiet=False, fuzzy=True)
    
    # url = 'https://drive.google.com/uc?id=1-JWdjBaHJTB5i5WskBBlSCJT0EHDIlX3'
    # output = '20150428_collected_images.tgz'
    # gdown.download(url, output, quiet=False)
    
    url_train = "https://drive.google.com/drive/u/0/folders/1-HIoQo7XbOB2EX5ItsW0i9v0aayDp4kM"
    url_val = "https://drive.google.com/drive/u/0/folders/1-QC5N7sBWTmJsa9poXtlZPJFp5bZ1Fo6"
    url_test = "https://drive.google.com/drive/u/0/folders/1-JWdjBaHJTB5i5WskBBlSCJT0EHDIlX3"
    
    gdown.download_folder(url=url_train)
    gdown.download_folder(url=url_val)
    gdown.download_folder(url=url_test)