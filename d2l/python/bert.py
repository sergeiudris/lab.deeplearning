import gluonnlp as nlp; import mxnet as mx;

# will be in /root/.mxnet/models
model, vocab = nlp.model.get_model('bert_12_768_12', dataset_name='book_corpus_wiki_en_uncased', use_classifier=False, use_decoder=False);
tokenizer = nlp.data.BERTTokenizer(vocab, lower=True);
transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=512, pair=False, pad=False);
sample = transform(['Hello world!']);
words, valid_len, segments = mx.nd.array([sample[0]]), mx.nd.array([sample[1]]), mx.nd.array([sample[2]]);
seq_encoding, cls_encoding = model(words, segments, valid_len);