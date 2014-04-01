#!/usr/bin/env python

import bs4
import sys

output_buf = []

from lxml import etree
root = etree.Element("ReutersDataset")

for input_file in sys.argv[1:]:

    soup = bs4.BeautifulSoup(open(input_file, 'r'))
    for reuter in soup.find_all('reuters'):
        # Output XML element
        current = etree.Element("Reuters")
        # Extract training/test
        train = reuter['lewissplit']
        current.set('lewissplit',train)

        # Count the number of topics
        count = 0
        topics_elem = etree.Element("Topics")
        for topic in reuter.find('topics').findAll(text=True):
            topic_elem = etree.Element("Topic")
            topic_elem.text = topic
            topics_elem.append(topic_elem)
            count += 1
        if count == 0:
            continue
        current.append(topics_elem)

        title_elem = etree.Element("Title")
        title_tag = reuter.find('title')
        if title_tag != None:
            title_elem.text = title_tag.text
            current.append(title_elem)

        for a in reuter.find('text').children:
            if (isinstance(a, bs4.element.Tag)):
                a.decompose()
        text_tag = etree.Element('Text')
        text_tag.text = etree.CDATA(reuter.find('text').text)
        current.append(text_tag)
        root.append(current)

print etree.tostring(root)
