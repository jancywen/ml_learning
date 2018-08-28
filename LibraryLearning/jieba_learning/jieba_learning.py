# -*- coding: utf-8 -*-

import jieba

sentence = "接天莲叶无穷碧,映日荷花别样红"

# 精准模式和全模式就是 多项式和伯努利的区别 默认精准模式
# 精准模式
c1 = jieba.cut(sentence=sentence, cut_all=False)
# 全模式
c2 = jieba.cut(sentence, cut_all=True)
# 关闭隐马尔科夫模式
c3 = jieba.cut(sentence, cut_all=False, HMM=False)


cs = jieba.cut_for_search(sentence)

lc = jieba.lcut(sentence)

lcs = jieba.lcut_for_search(sentence=sentence)

# print("/".join(c1))
# print("*".join(c2))
# print("--".join(c3))
# print(" ".join(cs))
#
# print(lc)
# print(c1)

# 词性
import jieba.posseg as pg

pgs = pg.cut(sentence)
print(pgs)
for w in pgs:
    print(w.word, end='')
    print(w.flag)
    print(w)


# 加载自定义 字典分词
# 词典格式和 dict.txt 一样，一个词占一行；每一行分三部分：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒。
# file_name 若为路径或二进制方式打开的文件，则文件必须为 UTF-8 编码。
jieba.load_userdict("./dict.txt")


pgss = pg.lcut(sentence)

print(pgss)

# 动态调整字典
# 添加分词
jieba.add_word("无穷碧")
jieba.del_word("")

print('*'*40)
testlist = [
('今天天气不错', ('今天', '天气')),
('如果放到post中将出错。', ('中', '将')),
('我们中出了一个叛徒', ('中', '出')),
]

for sent, seg in testlist:
    print('/'.join(jieba.cut(sent, HMM=False)))
    word = ''.join(seg)
    print('%s Before: %s, After: %s' % (word, jieba.get_FREQ(word), jieba.suggest_freq(seg, True)))
    print('/'.join(jieba.cut(sent, HMM=False)))
    print("-"*40)


