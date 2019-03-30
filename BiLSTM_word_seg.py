# -*- coding: utf-8 -*-
import re
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import GRU, Embedding, Bidirectional, TimeDistributed, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

# 训练数据路径
TRAIN_DATA_PATH = './train_data/msr_training.utf8'
DATA_INFOR_PATH = './model_file/my_data_infor.pkl'
TOKENIZER_PATH = './model_file/my_tokenizer.pkl'
SEQ_PATH = './training/my_train_seq.np'
LABEL_PATH = './training/my_train_label.np'
CHECKPOINT_PATH = './model_file/my_model.hdf5'
TENSORBOARD_PATH = './model_file/my_tensorboard'


# 用于匹配的模式对象
# \u4e00-\u9fa5表示所有中文，20901个汉字
# 本处表示不匹配汉字和字母
reOBJ = re.compile('[^\u4e00-\u9fa5a-z\s]')

# 字标注序列
TAG = {'S': 1, 'B': 2, 'M': 3, 'E': 4}

# 最大句子长度，不足补0
MAX_SENTENCE = 50

VOCAB_SIZE = 5033
EMBEDDING_OUT_DIM = 64
HIDDEN_UNITS = 200
DROPOUT_RATE = 0.3
NUM_CLASS = 5

# 整理训练数据格式
class TrainDataProcess():
    def __init__(self):
        self.seq = []
        self.label = []
        
    # 读取原始数据并整理格式，转换成list返回
    def loadOriginData(self, fileName):

        # 清除不匹配的引号
        def clearQuotation(data):
            if '“' not in data:
                data = data.replace('”', '')
            elif '”' not in data:
                data = data.replace('“', '')

            if '‘' not in data:
                data = data.replace('’', '')
            elif '’' not in data:
                data = data.replace('‘', '')
            
            return data.lower()

        f = open(fileName, encoding='utf8')
        d = f.read().split('\n')
        
        return list(map(clearQuotation, d))
    
    # 读取整理过的语料，建立训练数据
    def buildTrainData(self, origindata):
        def tagWord(word):
            length = len(word)
            if length == 0:
                return []
            elif length == 1:
                return ['S']
            elif length == 2:
                return ['B','E']
            else:
                return ['B']+['M']*(length-2)+['E']

        x = []
        y = []
        # 获取原始数据中的一行文本
        for oneline in origindata:
            # 将这一行文本按照标点符号分割成多个句子
            sents = re.split(reOBJ, oneline)
            # 对每个句子按照空格划分成多个词
            for sent in sents:
                words = sent.split()
                tmp_x = []
                tmp_y = []
                for word in words:
                    tmp_x.extend(list(word))
                    tmp_y.extend(tagWord(word))
                if 1 <= len(tmp_x) <= MAX_SENTENCE:
                    x.append(' '.join(tmp_x))
                    y.append(' '.join(tmp_y))
        return x, y
        
    # 处理数据的主控函数，将整理过的数据转换成填充好的序列
    def createTrainData(self):
        origindata = self.loadOriginData(TRAIN_DATA_PATH)
        train_x, train_y = self.buildTrainData(origindata)
        
        # 统计总字数，使用set除去重复
        wordnum = len(set(' '.join(train_x).split()))
        # 向量化文本
        tokenizer = Tokenizer(num_words=wordnum)
        # 设置要训练的文本，建立序列字典
        tokenizer.fit_on_texts(train_x)
        # 将文本转换成对应的序列并填充到MAX_SENTENCE
        self.seq = tokenizer.texts_to_sequences(train_x)
        self.seq = pad_sequences(self.seq, maxlen=MAX_SENTENCE)

        # 将原本字标注转换成序列并填充
        pad_label = []
        for i in train_y:
            # i 格式 ‘S S B E S S’
            tag = i.split()
            tmp = np.zeros((len(tag), 5))
            # 0表示填充字符的序列
            tmp2 = np.array([[1, 0, 0, 0, 0]] * (MAX_SENTENCE))
            for idx, j in enumerate(tag):
                tmp[idx][TAG.get(j, 0)] = 1
            tmp2[-len(tag):] = tmp
            pad_label.append(tmp2)
        self.label = np.array(pad_label)

        # 保存训练数据，seq为输入，label为标签
        self.seq.dump(SEQ_PATH)
        self.label.dump(LABEL_PATH)

        # 保存分类器
        data_infor = {'max_len': MAX_SENTENCE, 'num_words': wordnum}
        pickle.dump(data_infor, open(DATA_INFOR_PATH, 'wb'))
        pickle.dump(tokenizer, open(TOKENIZER_PATH, 'wb'))


class Train():
    def __init__(self):
        self.model = Sequential([
            Embedding(VOCAB_SIZE + 1, EMBEDDING_OUT_DIM, mask_zero=True),
            Bidirectional(GRU(HIDDEN_UNITS // 2, return_sequences=True)),
            TimeDistributed(Dense(NUM_CLASS, activation='softmax'))
        ])

        self.model.summary()
        self.model.compile(optimizer=Adam(),loss='categorical_crossentropy',metrics=['accuracy'])

    def doTrain(self):
        seq = np.load(SEQ_PATH)
        label = np.load(LABEL_PATH)
        checkpoint = ModelCheckpoint(
            CHECKPOINT_PATH,
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='max',
            period=1
        )
        tensorboard = TensorBoard(log_dir=TENSORBOARD_PATH)

        callbacklist = [checkpoint, tensorboard]
        
        self.model.fit(
            seq, label,
            batch_size=128,
            epochs=20,
            validation_split=0.2,
            callbacks=callbacklist
        )

class Seg():
    def __init__(self):
        # 载入模型
        self.model = Sequential([
            Embedding(VOCAB_SIZE + 1, EMBEDDING_OUT_DIM, mask_zero=True),
            Bidirectional(GRU(HIDDEN_UNITS // 2, return_sequences=True)),
            TimeDistributed(Dense(NUM_CLASS, activation='softmax'))
        ])

        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.load_weights(CHECKPOINT_PATH)

        self.tokenizer = pickle.load(open(TOKENIZER_PATH,'rb'))
        # 转移概率
        # self.trans = {'SS': 0.46460125869921165,
        #     'SB': 0.3503213832274479,
        #     'BM': 0.1167175117318146,
        #     'BE': 0.8832824882681853,
        #     'MM': 0.2777743117140081,
        #     'ME': 0.7222256882859919,
        #     'ES': 0.5310673430644739,
        #     'EB': 0.46893265693552616
        # }
        self.trans = {'SS': 0.5, 'SB': 0.5, 'BM': 0.5, 'BE': 0.5, 'MM': 0.5, 'ME': 0.5, 'ES': 0.5, 'EB': 0.5}
        self.trans = {i: np.log(self.trans[i]) for i in self.trans.keys()}

    # 分词
    def cut(self, doc):
        divi = re.compile('[^\u4e00-\u9fa5\s]')
        sentences = re.finditer(divi, doc)
        result = []
        i = 0
        for sent in sentences:
            one_sent = doc[i:sent.start()]
            result.extend(self.getWord(one_sent))
            result.extend(doc[sent.start():sent.end()])
            i = sent.end()
        result.extend(self.getWord(doc[i:]))
        return result

    def getWord(self, one_sent):
        if not one_sent:
            return []
        else:
            words = self.tokenizer.texts_to_sequences([list(one_sent),])
            pad_words = pad_sequences(words, maxlen=MAX_SENTENCE)
            predict = self.model.predict(pad_words)
            predict = predict[0][-len(words[0]):]

            pre_tags = [dict(zip(['S','B', 'M', 'E'], i[1:])) for i in predict]

            final_tags = self.viterbi(pre_tags)
            final_words = []
            x = 0
            for i in range(len(one_sent)):
                if not self.tokenizer.texts_to_sequences(one_sent[i])[0]:
                    final_words.append(one_sent[i])
                    x += 1
                    continue
                if final_tags[i-x] in ['S', 'B']:
                    final_words.append(one_sent[i-x])
                else:
                    final_words[-1] += one_sent[i-x]
            return final_words

    def viterbi(self, pre_tags):
        last = {'S': pre_tags[0]['S'], 'B': pre_tags[0]['B']}
        for i in range(1, len(pre_tags)):
            last1 = last.copy()
            last = {}
            for j in pre_tags[i].keys():
                now = {}
                for k in last1.keys():
                    if k[-1] + j in self.trans.keys():
                        now[k + j] = last1[k] + pre_tags[i][j] + self.trans[k[-1] + j]
                m = max(now.items(), key=lambda x: x[1])
                last[m[0]] = m[1]
        return max(last.items(),key = lambda x:x[1])[0]


if __name__ == '__main__':
    # TrainDataProcess().createTrainData()
    # Train().doTrain()

    seg = Seg()
    print(seg.cut('扬帆远东做与中国合作的先行'))
    print(seg.cut('希腊的经济结构较特殊。'))
    print(seg.cut('海运业雄踞全球之首，按吨位计占世界总数的１７％。'))
    print(seg.cut('另外旅游、侨汇也是经济收入的重要组成部分，制造业规模相对较小。'))
    print(seg.cut('多年来，中希贸易始终处于较低的水平，希腊几乎没有在中国投资。'))
    print(seg.cut('十几年来，改革开放的中国经济高速发展，远东在崛起。'))
    print(seg.cut('瓦西里斯的船只中有４０％驶向远东，每个月几乎都有两三条船停靠中国港口。'))
    print(seg.cut('他感受到了中国经济发展的大潮。'))
    # print(seg.cut('一个人总是有极限的，所以，jojo我不做人了'))
    # print(seg.cut('你们呀，不要听风就是雨'))
    # print(seg.cut('我算是见多识广了，西方哪个国家我没去过'))
    # print(seg.cut('我乔鲁诺·乔巴拿有一个自认为是正确的梦想，那就是成为秧歌star'))
    # print(seg.cut('南京市长在南京市长江大桥发表讲话'))
    # print(seg.cut('还好我一把把把把住了'))
