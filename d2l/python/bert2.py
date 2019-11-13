import warnings
warnings.filterwarnings('ignore')

import io
import random
import numpy as np
import mxnet as mx
import gluonnlp as nlp
# from bert import data, model

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
# change `ctx` to `mx.cpu()` if no GPU is available.
ctx = mx.cpu(0)

bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
# print(bert_base)
# print(bert_base.export)
bert_base.hybridize()
bert_base(mx.nd.ones((1,512,768)))
bert_base.export("/opt/app/tmp/data/models/bert_base_py", epoch=0)
