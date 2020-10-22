import pymysql
import traceback
import xml.etree.ElementTree as ET
import re
import os
import sys

data_dir = './../dataset/stackoverflow'

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
