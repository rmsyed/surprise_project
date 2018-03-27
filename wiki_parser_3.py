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




# protects against invalid unicode characters
def rectify(selstr):
	val = ''.join([x for x in selstr if ord(x)<128])
	return str(val)



stemmer = PorterStemmer()
stops = set(stopwords.words("english"))

def tokenizer(text, token_min_len,token_max_len, lower):
	fullsents = nltk.sent_tokenize(text)
	fullseqs = []
	for sent in fullsents:
		fullseqs += [rectify(term.lower()) for term in nltk.word_tokenize(sent) if ((term not in stops) and term.isalnum())]
	fullseqs_stemmed = [stemmer.stem(term) for term in fullseqs] #... keep both the stemmed and unstemmed versions
	return [fullseqs, fullseqs_stemmed]


def loadSimpleWiki():
	#file_name = "C:/Users/Admin/Anaconda2/envs/py27/corpora/wiki2017/simplewiki-20170820-pages-meta-current.xml.bz2"
	#counter  =0
	#... 179,620 articles found
	file_name = "C:/Users/Admin/Anaconda2/envs/py27/corpora/wiki2017/simplewiki-20170820-pages-meta-current.xml.bz2"

	wiki = WikiCorpus(file_name, lemmatize=False, dictionary={}) #vocab dict not needed
	i = 0
	#for text in wiki.get_texts():
	allwords = defaultdict(lambda:0)
	allsents = []
	print ("starting...")
	dfcounter_stemmed = defaultdict(lambda: 0)
	dfcounter_nostem = defaultdict(lambda: 0)

	for (title,article,pageid) in extract_pages(bz2.BZ2File(file_name), filter_namespaces=('0', )): #filter_namespaces=["0"]):
		if len(article)==0:
			continue

		text = process_article((filter_wiki(article), False, title, pageid), tokenizer_func=tokenizer)
		text_orig = set(text[0][0]) #... We are ONLY interested in whether or not the term appeared in this document NOT how many times or where
		text_stemmed = set(text[0][1])

		for term in text_orig:
			dfcounter_nostem[term] += 1
		for term in text_stemmed:
			dfcounter_stemmed[term] += 1

		i+=1
		if i%1000==0:
			print (i)
			#break
	print (i)

	handle = open("simplewiki_docfreqs_stemmed.txt","w+")
	for key,val in dfcounter_stemmed.items():
		handle.write(str(key)+"\t"+str(val)+"\n")
	handle.close()

	handle = open("simplewiki_docfreqs_nostem.txt","w+")
	for key,val in dfcounter_nostem.items():
		handle.write(str(key)+"\t"+str(val)+"\n")
	handle.close()


if __name__ == "__main__":
	loadSimpleWiki()
	sys.exit()










#........... Get document frequencies from the BOW files generated through the gensim script...........
#.......................................................................................................
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


#documentFrequencies()
