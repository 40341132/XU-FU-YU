from bs4 import BeautifulSoup
import requests
from time import sleep
import re
import json

html = requests.get("http://www.example.com").text
soup = BeautifulSoup(html, 'html5lib')

tds = soup('td', 'thumbtext')
def is_video(td):
 """it's a video if it has exactly one pricelabel, and if
Scraping the Web | 111
www.it-ebooks.info
 the stripped text inside that pricelabel starts with 'Video'"""
 pricelabels = td('span', 'pricelabel')
 return (len(pricelabels) == 1 and
 pricelabels[0].text.strip().startswith("Video"))
print len([td for td in tds if not is_video(td)])
# 21 for me, might be different for you

def book_info(td):
 """given a BeautifulSoup <td> Tag representing a book,
 extract the book's details and return a dict"""
 title = td.find("div", "thumbheader").a.text
 by_author = td.find('div', 'AuthorName').text
 authors = [x.strip() for x in re.sub("^By ", "", by_author).split(",")]
 isbn_link = td.find("div", "thumbheader").a.get("href")
 isbn = re.match("/product/(.*)\.do", isbn_link).groups()[0]
 date = td.find("span", "directorydate").text.strip()
 return {
 "title" : title,
 "authors" : authors,
 "isbn" : isbn,
 "date" : date
 }

from time import sleep
base_url = "http://shop.oreilly.com/category/browse-subjects/" + \
 "data.do?sortby=publicationDate&page="
books = []
NUM_PAGES = 2 # at the time of writing, probably more by now
for page_num in range(1, NUM_PAGES + 1):
 print "souping page", page_num, ",", len(books), " found so far"
 url = base_url + str(page_num)
 soup = BeautifulSoup(requests.get(url).text, 'html5lib')
 for td in soup('td', 'thumbtext'):
    if not is_video(td):
     books.append(book_info(td))
 # now be a good citizen and respect the robots.txt!
sleep(2)
print books



serialized = """{ "title" : "Data Science Book",
 "author" : "Joel Grus",
 "publicationYear" : 2014,
 "topics" : [ "data", "science", "data science"] }"""
# parse the JSON to create a Python dict
deserialized = json.loads(serialized)
if "data science" in deserialized["topics"]:
 print deserialized

endpoint = "https://api.github.com/users/40341132/repos"
repos = json.loads(requests.get(endpoint).text)
for rows in repos:
    data1 = rows["name"]
    print(data1)