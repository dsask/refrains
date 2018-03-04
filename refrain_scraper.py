# -*- coding: utf-8 -*-

import re
from requests import get

import pandas as pd
import unicodecsv as csv
from bs4 import BeautifulSoup

## Scrapes all refrains from refrain.ac.uk.
#
# Return a list of dictionaries containing:
#    'music_link' : a url to the image of the sheet music for the refrain.
#    'manuscript' : name of the manuscript the refrain is from.
#    'parent_work' : name of the work the refrain is from.
#
# Example usage:
#   complete_url = get_refrain_page_links()
#   all_metadata = scrape_all_refrain_metadata()

def get_refrain_page_links():
    url = 'http://refrain.ac.uk/view/abstract_item/'
    response = get(url)

    data = response.text

    soup = BeautifulSoup(data)

    links = []
    for link in soup.find_all('a'):
         links.append(link.get('href'))
    m = [bool(re.match('^[0-9].*html', s)) for s in links]
    refrain_links = pd.Series(links)[m]
    refrain_links = refrain_links.reset_index()

    complete_urls = []
    for r in refrain_links.index:
        complete_urls.append(url + refrain_links.loc[r,].astype(str).values[1])
    return complete_urls

def scrape_all_refrain_metadata(complete_urls):
    all_metadata = []
    for u in complete_urls:
        all_metadata = get_metadata(u, all_metadata)
    return all_metadata

def get_metadata(this_url, all_metadata):
    this_response = get(this_url)
    this_data = this_response.text
    this_soup = BeautifulSoup(this_data)
    refrain_boxes = this_soup.findAll("div", {"class": "ep_summary_box"})
    for b in refrain_boxes:
        metadata = extract_refrain_box_data(b)
        metadata['url'] = this_url
        metadata['vdb_number'] = get_vdb_number(this_url)
        all_metadata.append(metadata)
    return(all_metadata)

def extract_refrain_box_data(refrain_box):
    metadata = {
        'music_link' : find_music_link(refrain_box),
        'manuscript' : find_source_manuscript(refrain_box),
        'parent_work' : find_parent_work_title(refrain_box)
    }
    return metadata

def find_music_link(refrain_box):
    music_links = refrain_box.findAll("img", {"class": "music"})
    if len(music_links)==1:
        s = str(music_links)
        split_string = str(s).split('"')
        music_links = split_string[-2]
    return music_links

def find_source_manuscript(refrain_box):
    table = refrain_box.find(lambda tag: tag.name=='table')
    rows = table.findAll(lambda tag: tag.name=="tr")
    for r in rows:
        if r.find(lambda tag: tag.name=="th").text == "Manuscript:":
            return r.find(lambda tag: tag.name=="td").text
        else:
            return None

def find_parent_work_title(refrain_box):
    for div in refrain_box.findAll("div", {"class": "ep_no_js"}):
        if div.text[:len("Parent Work: ")] == "Parent Work: ":
            return div.text[len("Parent Work: "):]

def print_output_csv(all_metadata, output_filename = "refrain_metadata.csv"):
    metadata_list = []
    for metadata in all_metadata:
        if metadata['manuscript'] is None:
            continue
        else:
            manuscript = unicode(metadata['manuscript']).encode('utf-8')
        if metadata['parent_work']:
            parent_work = unicode(metadata['parent_work']).encode('utf-8')
        else:
            parent_work = None
        metadata_list.append(
            [metadata['vdb_number'],
             metadata['url'],
             metadata['manuscript'],
             parent_work,
             metadata['music_link']
            ]
        )
    with open(output_filename, 'wb') as f:
        writer = csv.writer(f, encoding='utf-8')
        writer.writerows(metadata_list)

## Helper fucntions ##

def get_vdb_number(url):
    return "vdB" + url.split("/")[-1].split(".html")[0]
