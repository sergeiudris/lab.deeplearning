
Dive into Deep Learning
Release 0.7
Aston Zhang, Zack C. Lipton, Mu Li, Alex J. Smola
Oct 14, 2019

## links

- https://d2l.ai/

## notes

- 11
  - > Machine learning (ML) is the study of
    powerful techniques that can learn behavior from experience. As ML algorithm accumulates more experience,
    typically in the form of observational data or interactions with an environment, their performance improves.

- 14
  - > Deep models are deep in precisely
    the sense that they learn many layers of computation. It turns out that these many-layered (or hierarchical)
    models are capable of addressing low-level perceptual data in a way that previous tools could not. In bygone
    days, the crucial part of applying ML to these problems consisted of coming up with manually engineered
    ways of transforming the data into some form amenable to shallow models. One key advantage of deep
    learning is that it replaces not only the shallow models at the end of traditional learning pipelines, but also
    the labor-intensive feature engineering. Secondly, by replacing much of the domain-specific preprocessing,
    deep learning has eliminated many of the boundaries that previously separated computer vision, speech
    recognition, natural language processing, medical informatics, and other application areas, offering a unified
    set of tools for tackling diverse problems.
    
- 15
  - > Earlier, we introduced machine learning as “learning behavior from experience”. By learning here, we mean
      improving at some task over time. But who is to say what constitutes an improvement?
    > In order to develop a formal mathematical system of learning machines, we need to have formal measures
      of how good (or bad) our models are. In machine learning, and optimization more generally, we call these
      objective functions.
      By convention, we usually define objective funcitons so that lower is better.
      Because lower is better, these functions are sometimes called `loss` functions or `cost` functions.

- 16
  - > • Training Error: The error on that data on which the model was trained. You could think of this as
        being like a student’s scores on practice exams used to prepare for some real exam. Even if the results
        are encouraging, that does not guarantee success on the final exam.

    > • Test Error: This is the error incurred on an unseen test set. This can deviate significantly from the
        training error. When a model fails to generalize to unseen data, we say that it is overfitting. In real-life
        terms, this is like flunking the real exam despite doing well on practice exams.

- 21
  - > At the
      National Library of Medicine, a number of professional annotators go over each article that gets indexed
      in PubMed to associate it with the relevant terms from MeSH, a collection of roughly 28k tags. This is a
      time-consuming process and the annotators typically have a one year lag between archiving and tagging.
      Machine learning can be used here to provide provisional tags until each article can have a proper manual
      review. Indeed, for several years, the BioASQ organization has hosted a competition 14 to do precisely this.

- 25
  - supervised
    - regression
    - classification
    - tagging
    - search and ranking
    - recommend
    - sequence learning
      - tagging and parsing
      - auto speach recognition
      - text to speech
      - machine translation

  - unsupervised
    - clustering
    - subspace estimation (when dependence is linear = principal component analysis)
    - representation learning (Rome - Italy + France = Paris)
    - directed graphical models, causality
    - generative adversarial networks (synthesize data)
  
  - reinforcement
  
- 136
  - > The phenomena of fitting our training data more closely than we fit the underlying distribution is called
      overfitting, and the techniques used to combat overfitting are called regularization

- 138
  - > When we train our models, we attempt are searching for a function that fits the training data as well as
      possible. If the function is so flexible that it can catch on to spurious patterns just as easily as to the true
      associations, then it might peform too well without producing a model that generalizes well to unseen data.
      This is precisely what we want to avoid (or at least control). Many of the techniques in deep learning are
      heuristics and tricks aimed at guarding against overfitting.

  - > In this chapter, to give you some intuition, we’ll focus on a few factors that tend to influence the generaliz-
      ability of a model class:
      1. The number of tunable parameters. When the number of tunable parameters, sometimes called the
      degrees of freedom, is large, models tend to be more susceptible to overfitting.
      2. The values taken by the parameters. When weights can take a wider range of values, models can be
      more susceptible to over fitting.
      3. The number of training examples. It’s trivially easy to overfit a dataset containing only one or two
      examples even if your model is simple. But overfitting a dataset with millions of examples requires an
      extremely flexible model.

- 140
  - > Moreover, in general,
      more data never hurts. For a fixed task and data distribution, there is typically a relationship between
      model complexity and dataset size. Given more data, we might profitably attempt to fit a more complex
      model. Absent sufficient data, simpler models may be difficult to beat. For many tasks, deep learning only
      outperforms linear models when many thousands of training examples are available. In part, the current
      success of deep learning owes to the current abundance of massive datasets due to internet companies, cheap
      storage, connected devices, and the broad digitization of the economy.

- 154
  - > Their proposed idea is called dropout, and it is now a standard technique that is widely used
      for training neural networks. Throughout trainin, on each iteration, dropout regularization consists simply
      of zeroing out some fraction (typically 50%) of the nodes in each layer before calculating the subsequent
      layer.

  - > Intuitively,
      deep learning researchers often explain the inutition thusly: we do not want the network’s output to depend
      too precariously on the exact activation pathway through the network. The original authors of the dropout
      technique described their intuition as an effort to prevent the `co-adaptation` of feature detectors.

- 328
  - > The key distinction between regular RNNs and GRUs is that the latter support gating of the hidden state.
      This means that we have dedicated mechanisms for when the hidden state should be updated and also when
      it should be reset.

- 565 
  - > As its name implies, a word vector is a vector used to represent a
      word. It can also be thought of as the feature vector of a word. The technique of mapping words to vectors
      of real numbers is also known as word embedding.

