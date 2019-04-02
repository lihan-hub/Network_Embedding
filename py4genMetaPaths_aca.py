# -*- coding: utf-8 -*-
# coding: utf-8
import sys
import os
import random
from collections import Counter

class MetaPathGenerator:
	def __init__(self):
		self.id_author = dict()
		self.id_conf = dict()
		self.author_coauthorlist = dict()
		self.conf_authorlist = dict()
		self.author_conflist = dict()
		self.paper_author = dict()
		self.author_paper = dict()
		self.conf_paper = dict()
		self.paper_conf = dict()

	def read_data(self, dirpath):
		with open(dirpath + "/id_author.txt", encoding="unicode_escape") as adictfile:
			for line in adictfile:
				toks = line.strip().split("\t")
				if len(toks) == 2:
					self.id_author[toks[0]] = toks[1].replace(" ", "")	# 一个id对应一个作者

		#print "#authors", len(self.id_author)

		with open(dirpath + "/id_conf.txt", encoding="unicode_escape") as cdictfile:
			for line in cdictfile:
				toks = line.strip().split("\t")
				if len(toks) == 2:
					newconf = toks[1].replace(" ", "")
					self.id_conf[toks[0]] = newconf	# 一个id对应一场会议

		#print "#conf", len(self.id_conf)

		with open(dirpath + "/paper_author.txt", encoding="unicode_escape") as pafile:
			for line in pafile:
				toks = line.strip().split("\t")
				if len(toks) == 2:
					p, a = toks[0], toks[1]
					if p not in self.paper_author:
						self.paper_author[p] = []	# 一篇论文对应多个作者
					self.paper_author[p].append(a)	
					if a not in self.author_paper:
						self.author_paper[a] = []	# 一个作者对应多篇论文
					self.author_paper[a].append(p)

		with open(dirpath + "/paper_conf.txt", encoding="unicode_escape") as pcfile:
			for line in pcfile:
				toks = line.strip().split("\t")
				if len(toks) == 2:
					p, c = toks[0], toks[1]
					self.paper_conf[p] = c 			# 一篇论文对应一场会议
					if c not in self.conf_paper:
						self.conf_paper[c] = []		# 一场会议对应多篇论文
					self.conf_paper[c].append(p)

		sumpapersconf, sumauthorsconf = 0, 0
		conf_authors = dict()
		for conf in self.conf_paper:
			papers = self.conf_paper[conf]
			sumpapersconf += len(papers)			# 总的会议论文数
			for paper in papers:
				if paper in self.paper_author:
					authors = self.paper_author[paper]
					sumauthorsconf += len(authors)	# 总的会议作者数

		print("#confs  ", len(self.conf_paper))
		print("#papers ", sumpapersconf,  "#papers per conf ", sumpapersconf / len(self.conf_paper)) 
		print("#authors", sumauthorsconf, "#authors per conf", sumauthorsconf / len(self.conf_paper)) 


	def generate_random_aca(self, outfilename, numwalks, walklength):
		for conf in self.conf_paper:
			self.conf_authorlist[conf] = []			# 每场会议的作者列表
			for paper in self.conf_paper[conf]:
				if paper not in self.paper_author: continue
				for author in self.paper_author[paper]:
					self.conf_authorlist[conf].append(author)	# 添加作者至当前会议作者列表中
					if author not in self.author_conflist:
						self.author_conflist[author] = []		# 一个作者对应多个会议
					self.author_conflist[author].append(conf)
		#print "author-conf list done"

		outfile = open(outfilename, 'w', encoding="utf-8")
		for conf in self.conf_authorlist:
			conf0 = conf
			for j in range(0, numwalks ): #wnum walks
				outline = self.id_conf[conf0]
				for i in range(0, walklength):					# walklength与numa以及numc之间的数值差异是否有待研究？
					authors = self.conf_authorlist[conf]
					numa = len(authors)
					authorid = random.randrange(numa)			# 返回(0,numa)区间的随机数，直接用于作者id，是否存在问题？
					author = authors[authorid]					# 返回的随机数并不是作为作者id，而是对应会议作者列表中的列表索引
					outline += " " + self.id_author[author]
					confs = self.author_conflist[author]
					numc = len(confs)
					confid = random.randrange(numc)
					conf = confs[confid]
					outline += " " + self.id_conf[conf]
				outfile.write(outline + "\n")
		outfile.close()


#python py4genMetaPaths.py 1000 100 net_aminer output.aminer.w1000.l100.txt
#python py4genMetaPaths.py 1000 100 net_dbis   output.dbis.w1000.l100.txt


dirpath = "net_aminer" 
# OR 
dirpath = "net_dbis"

numwalks = int(sys.argv[1])
walklength = int(sys.argv[2])

dirpath = sys.argv[3]
outfilename = sys.argv[4]

def main():
	mpg = MetaPathGenerator()
	mpg.read_data(dirpath)
	mpg.generate_random_aca(outfilename, numwalks, walklength)


if __name__ == "__main__":
	main()



'''
dirpath = "net_dbis"
outfilename = "output.dbis.w1000.l100.txt"
numwalks = 1000
walklength = 100

mpg = MetaPathGenerator()
mpg.read_data(dirpath)
mpg.generate_random_aca(outfilename, numwalks, walklength)
'''


























