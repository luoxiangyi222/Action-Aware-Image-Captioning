"""
Author: Xiangyi Luo

Parsing information of stackoverflow dump
"""

import xml.etree.ElementTree as ET
import re
import csv

data_path = './../dataset/stackoverflow/Posts.xml'

parsed_data = []
# id, type, parent-id, score, text, code
parsed_data_path = 'parsed_stackoverflow.csv'
parsed_data_file = open(parsed_data_path, 'w+')
wr = csv.writer(parsed_data_file, quoting=csv.QUOTE_ALL)
wr.writerow(['id', 'type', 'parent_id', 'score', 'text', 'code'])

i = 0
for event, elem in ET.iterparse(data_path):
    if i < 10000:
        if elem.tag == 'row':

            if elem.attrib['PostTypeId'] == '1' and re.search('android', elem.attrib['Tags']):
                # print('=====')
                # print(elem.attrib)

                tags = re.findall(r'<.+?>', elem.attrib['Tags'])
                tags = ','.join(list(map(lambda x: x.replace('<', '').replace('>', ''), tags)))

                # print(tags)

                api = re.sub(r'-\d+\.*\d*', '', tags)
                api = list(set(re.split(r',|-|\.', api)))

                # print(api)

                if 'android' in api:
                    api.remove('android')
                    api = ','.join(api)

                    text = re.findall(r'<p>.+?</p>', elem.attrib['Body'], re.DOTALL)
                    text = ' '.join(list(map(lambda x: re.sub(
                        r'<.+?>|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', x), text)))

                    # print(text)

                    # methods = re.findall(r'\w+\(\)', text)
                    # methods = ','.join(list(set(map(lambda x: re.sub(r'\(\)', '', x), methods))))

                    code = re.findall(r'<pre><code>(.+?)</code></pre>', elem.attrib['Body'], re.DOTALL)
                    code = ' '.join(list(map(lambda x: re.sub(r'\s+', ' ', x), code)))

                    record = [elem.attrib['Id'], int(elem.attrib['PostTypeId']), -1, int(elem.attrib['Score']), text, code]
                    # parsed_data.append(record)
                    wr.writerow(record)

            elif elem.attrib['PostTypeId'] == '2':
                # print('=====')
                # print(elem.attrib)

                text = re.findall(r'<p>.+?</p>', elem.attrib['Body'], re.DOTALL)
                text = ' '.join(list(map(lambda x: re.sub(
                    r'<.+?>|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', x), text)))
                code = re.findall(r'<pre><code>(.+?)</code></pre>', elem.attrib['Body'], re.DOTALL)
                code = ' '.join(list(map(lambda x: re.sub(r'\s+', ' ', x), code)))

                record = [elem.attrib['Id'], int(elem.attrib['PostTypeId']), elem.attrib['ParentId'], int(elem.attrib['Score']), text, code]
                # parsed_data.append(record)
                wr.writerow(record)

        elem.clear()
        i = i + 1

parsed_data_file.close()

