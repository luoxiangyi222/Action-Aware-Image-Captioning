import pymysql
import traceback
import xml.etree.ElementTree as ET
import re
import os
import sys

data_dir = '/home/cheer/Project/Bridge/data/stackoverflow'

config = {
  'host': '172.17.0.2',
  'port': 3306,
  'user': 'root',
  'passwd':'123',
  'db': 'stackoverflow',
  'charset': 'utf8',
  #'cursorclass': pymysql.cursors.SSCursor
}

def main():
  connect = pymysql.connect(**config)
  cursor = connect.cursor()
  cursor.execute('DROP TABLE IF EXISTS question')
  cursor.execute('DROP TABLE IF EXISTS answer')
  cursor.execute('DROP TABLE IF EXISTS comment')

  cursor.execute('CREATE TABLE question (Id int(11) primary key, AcceptedAnswerId int(11), Score int(11), Tags varchar(1000), API varchar(200), Method varchar(1000), Link varchar(1000), Text text, Code text)')
  cursor.execute('CREATE TABLE answer (Id int(11) primary key, ParentId int(11), Score int(11), Link varchar(1000), Text text, Code text)')
  cursor.execute('CREATE TABLE comment (Id int(11) primary key, PostId int(11), Score int(11), Link varchar(1000), Text text)')

  for event, elem in ET.iterparse(os.path.join(data_dir, 'Posts.xml')):
    if elem.tag == 'row':
      sys.stdout.write('\r' + elem.attrib['Id'])
      sys.stdout.flush()
      if elem.attrib['PostTypeId'] == '1' and re.search('android', elem.attrib['Tags']):
        tags = re.findall(r'<.+?>', elem.attrib['Tags'])
        tags = ','.join(list(map(lambda x: x.replace('<', '').replace('>', ''), tags)))
        api = re.sub(r'-\d+\.*\d*', '', tags)
        api = list(set(re.split(r',|-|\.', api)))
        if 'android' in api:
          api.remove('android')
        api = ','.join(api)
        text = re.findall(r'<p>.+?</p>', elem.attrib['Body'], re.DOTALL)
        text = ' '.join(list(map(lambda x: re.sub(r'<.+?>|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', x), text)))
        methods = re.findall(r'\w+\(\)', text)
        methods = ','.join(list(set(map(lambda x: re.sub(r'\(\)', '', x), methods))))
        code = re.findall(r'<pre><code>(.+?)</code></pre>', elem.attrib['Body'], re.DOTALL)
        code = ' '.join(list(map(lambda x: re.sub(r'\s+', ' ', x), code)))
        link = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', elem.attrib['Body'])
        link = ','.join(list(map(lambda x: re.sub(r'<.+?>', '', x), link)))
        if len(link) > 1000:
          link = ''
        if 'AcceptedAnswerId' in elem.attrib.keys():
          cursor.execute('INSERT INTO question VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)', (elem.attrib['Id'], elem.attrib['AcceptedAnswerId'], elem.attrib['Score'], tags, api, methods, link, text, code))
        else:
          cursor.execute('INSERT INTO question VALUES (%s, null, %s, %s, %s, %s, %s, %s, %s)', (elem.attrib['Id'], elem.attrib['Score'], tags, api, methods, link, text, code))
        
      elif elem.attrib['PostTypeId'] == '2' and cursor.execute('SELECT * FROM question WHERE Id=%s', (elem.attrib['ParentId'])):
        text = re.findall(r'<p>.+?</p>', elem.attrib['Body'], re.DOTALL)
        text = ' '.join(list(map(lambda x: re.sub(r'<.+?>|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', x), text)))
        code = re.findall(r'<pre><code>(.+?)</code></pre>', elem.attrib['Body'], re.DOTALL)
        code = ' '.join(list(map(lambda x: re.sub(r'\s+', ' ', x), code)))
        link = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', elem.attrib['Body'])
        link = ','.join(list(map(lambda x: re.sub(r'<.+?>', '', x), link)))
        if len(link) > 1000:
          link = ''
        cursor.execute('INSERT INTO answer VALUES (%s, %s, %s, %s, %s, %s)', (elem.attrib['Id'], elem.attrib['ParentId'], elem.attrib['Score'], link, text, code))
    elem.clear()

  for event, elem in ET.iterparse(os.path.join(data_dir, 'Comments.xml')):
    if elem.tag == 'row':
      sys.stdout.write('\r' + elem.attrib['Id'])
      sys.stdout.flush()
      if cursor.execute('SELECT * FROM question WHERE Id=%s', (elem.attrib['PostId'])) or cursor.execute('SELECT * FROM answer WHERE Id=%s', (elem.attrib['PostId'])):
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', elem.attrib['Text'])
        link = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', elem.attrib['Text'])
        link = ','.join(list(map(lambda x: re.sub(r'<.+?>', '', x), link)))
        if len(link) > 1000:
          link = ''
        cursor.execute('INSERT INTO comment VALUES (%s, %s, %s, %s, %s)', (elem.attrib['Id'], elem.attrib['PostId'], elem.attrib['Score'], link, text))
    elem.clear()

  print (cursor.execute('SELECT * FROM question'), cursor.execute('SELECT * FROM answer'), cursor.execute('SELECT * FROM comment'))
  connect.commit()
  cursor.close()
  connect.close()

if __name__ == '__main__':
  main()
