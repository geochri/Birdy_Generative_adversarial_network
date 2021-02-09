# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 10:14:56 2018

@author: WT
"""

import re, os, time
from selenium import webdriver
 
from pattern.web import URL, DOM
import urllib.request
 
class GoogleImageExtractor(object):
 
    def __init__(self, search_key = '' ):
        if type(search_key) == str:
            self.g_search_key_list = [search_key]
        elif type(search_key) == list:
            self.g_search_key_list = search_key
        else:
            print('google_search_keyword not of type str or list')
            raise
        self.g_search_key = ''
        self.image_dl_per_search = 500
        self.prefix_of_search_url = "https://www.google.com.sg/search?q="
        self.postfix_of_search_url = '&source=lnms&tbm=isch&sa=X&ei=0eZEVbj3IJG5uATalICQAQ&ved=0CAcQ_AUoAQ&biw=939&bih=591'# non changable text
        self.target_url_str = ''
        self.pic_url_list = []
        self.pic_info_list = []
        self.folder_main_dir_prefix = "./data/"
 
    def reformat_search_for_spaces(self):
        self.g_search_key = self.g_search_key.rstrip().replace(' ', '+')
 
    def set_num_image_to_dl(self, num_image):
        self.image_dl_per_search = num_image
 
    def get_searchlist_fr_file(self, filename):
        with open(filename,'r') as f:
            self.g_search_key_list = f.readlines()
 
    def formed_search_url(self):
        self.reformat_search_for_spaces()
        self.target_url_str = self.prefix_of_search_url + self.g_search_key +\
                                self.postfix_of_search_url
 
    def retrieve_source_fr_html(self):
        try:
            driver = webdriver.Chrome()
        except:
            driver = webdriver.Firefox()
        driver.get(self.target_url_str)
        try:
            driver.execute_script("window.scrollTo(0, 30000)")
            time.sleep(2)
            self.temp_page_source = driver.page_source
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, 60000)")
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, 60000)")
 
        except:
            print('not able to find')
            driver.quit()
 
        self.page_source = driver.page_source
 
        driver.close()
 
    def extract_pic_url(self):
        dom = DOM(self.page_source)
        tag_list = dom('a.rg_l')
 
        for tag in tag_list[:self.image_dl_per_search]:
            tar_str = re.search('imgurl=(.*)&imgrefurl', tag.attributes['href'])
            try:
                self.pic_url_list.append(tar_str.group(1))
            except:
                print('error parsing', tag)
 
    def multi_search_download(self):
        for indiv_search in self.g_search_key_list:
            self.pic_url_list = []
            self.pic_info_list = []
 
            self.g_search_key = indiv_search
 
            self.formed_search_url()
            self.retrieve_source_fr_html()
            self.extract_pic_url()
            self.downloading_all_photos() 
            self.save_infolist_to_file()
 
    def downloading_all_photos(self):

        self.create_folder()
        pic_counter = 1
        for url_link in self.pic_url_list:
            url_link = re.sub('%3A', ":", url_link)
            url_link = re.sub('%2F', "/", url_link)
            print(pic_counter)
            pic_prefix_str = self.g_search_key  + str(pic_counter)
            self.download_single_image(url_link, pic_prefix_str)
            pic_counter = pic_counter +1
 
    def download_single_image(self, url_link, pic_prefix_str):

        self.download_fault = 0
        file_ext = os.path.splitext(url_link)[1] 
        #print(pic_prefix_str, file_ext)
        temp_filename = pic_prefix_str + str(file_ext)
        temp_filename_full_path = os.path.join(self.gs_raw_dirpath, temp_filename )
 
        valid_image_ext_list = ['.png','.jpg','.jpeg', '.gif', '.bmp', '.tiff'] #not comprehensive
 
        url = URL(url_link)
        if url.redirect:
            print("RD")
            return 
 
        if file_ext not in valid_image_ext_list:
            print("Invalid file type")
            return 
 
        f = open(temp_filename_full_path, 'wb')
        print(url_link)
        self.pic_info_list.append(pic_prefix_str + ': ' + url_link )
        try:
            urllib.request.URLopener.version = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134"
            #f.write(url.download(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.140 Safari/537.36 Edge/17.17134"))
            f.write(urllib.request.urlopen(url_link).read())
            #urllib.request.urlretrieve(url_link, temp_filename_full_path)
        except:
            print('Problem with processing this data: ', url_link)
            self.download_fault =1
        f.close()
 
    def create_folder(self):
        self.gs_raw_dirpath = os.path.join(self.folder_main_dir_prefix, time.strftime("_%d_%b%y", time.localtime()))
        if not os.path.exists(self.gs_raw_dirpath):
            os.makedirs(self.gs_raw_dirpath)
 
    def save_infolist_to_file(self):
        temp_filename_full_path = os.path.join(self.gs_raw_dirpath, self.g_search_key + '_info.txt' )
 
        with  open(temp_filename_full_path, 'w') as f:
            for n in self.pic_info_list:
                f.write(n)
                f.write('\n')
 
if __name__ == '__main__':
    w = GoogleImageExtractor('')
    searchlist_filename = "search_terms.txt"
    w.set_num_image_to_dl(500)
    w.get_searchlist_fr_file(searchlist_filename)#replace the searclist
    w.multi_search_download()