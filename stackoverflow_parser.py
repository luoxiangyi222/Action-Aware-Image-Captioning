"""
Author: Xiangyi Luo

Parsing information of stackoverflow dump
"""

import pymysql
import traceback
import xml.etree.ElementTree as ET
import re
import os
import sys

data_path = './../dataset/'