import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs
import os
import re
import csv

rootdir = '/Users/felixschekerka/Desktop/data/native_ads_data_unzipped'

def scrape_data(rootdir):
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if 'raw_html' in file:
                page = open(os.path.join(subdir, file), 'r')
                soup = bs(page, features="html.parser")

                # for script in soup(["script", "style"]):
                #     script.extract() 

                temp_text = []
                temp_title = []
                temp_id = []

                # print(os.path.join(subdir, file))

                text = soup.find_all('p')
                for element in text:
                    # ele = element.get_text().strip().replace("\r","").replace("\n","")
                    ele = element.get_text().strip(' \n\t')
                    result = re.sub('\\s+', ' ', ele)
                    if len(result) == 0 or result.count(' ')/len(result)>0.3:
                        pass
                    else:
                        temp_text.append(result)
                    # print(ele)
                
                title = soup.title

                if title == None:
                    # title = 'Title not existing'
                    # temp_title.append(title)
                    # print(title)
                    pass
                else:
                    title = title.get_text().strip(' \n\t')
                    temp_title.append(title)
                    # print(title.get_text().strip())

                temp_id.append(file)
                # print(file)

                rows = zip(temp_id, temp_title, temp_text)

                with open('extracted_data.csv', "a") as f:
                    writer = csv.writer(f)
                    for row in rows:
                        writer.writerow(row)

scrape_data(rootdir)