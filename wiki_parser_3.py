import cgi, cgitb, sys, os
from bs4 import BeautifulSoup, SoupStrainer, Comment
from collections import Counter, defaultdict
import shutil
import nltk
import random
import csv
import math
import numpy as np
from nltk.stem import *
from nltk.tokenize import RegexpTokenizer
import bz2
from bz2file import BZ2File
from gensim.corpora import WikiCorpus, MmCorpus, Dictionary
from gensim.corpora.wikicorpus import extract_pages, process_article



def documentFrequencies():
	dictionary = Dictionary.load_from_text('C:/Users/Admin/Anaconda2/envs/py27/corpora/wiki2017_wordids.txt.bz2')
	print (max(dictionary.token2id.values()))
	#... get the id corresponding to token "hello"
	tokenid= (dictionary.token2id["hello"])
	print (tokenid)
	#... get the document frequencies in the full corpus for which "hello" appeared
	print (dictionary.dfs[dictionary.token2id["hello"]])
	#... compute the total number of features in this corpus
	print (len(dictionary))


	#... CONSTRUCT THE Document Frequency OUTPUT FILE
	dforig = dictionary.dfs
	dfdict = {}
	for key,val in dforig.items():
		dfdict[str(dictionary[key])] = val

	fieldnames = ["term", "df"]
	with open("document_frequencies.tsv", "w+", encoding="utf-8") as handle:
		writer = csv.writer(handle, delimiter="\t")
		#writer.writerows(dfdict)
		for key, val in dfdict.items():
			writer.writerow([key,val])
		handle.close()







	#... load in the bag-of-words matrix market file for comparison
	mm_name = "C:/Users/Admin/Anaconda2/envs/py27/corpora/wiki2017_bow.mm"
	wikimodel = MmCorpus(mm_name)
	#... the matrix market should have the same number of features as len(dictionary)
	print (wikimodel)

	#... checked and verified that "hello" appears in the same number of documents as computed earlier
	if False:
		counter = 0
		featcount = 0
		for doc in wikimodel:
			res = [x for x in doc if x[0]==tokenid]
			if len(res)>0:
				#print (counter, ":",res)
				featcount +=1
				#break
		print (featcount)


documentFrequencies()
