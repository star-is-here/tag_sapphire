import re, json, pprint, time
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.mixture import DPGMM
from sklearn.pipeline import make_pipeline
from tqdm import *

pp = pprint.PrettyPrinter(indent=2)

# taken from sklearn example http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#example-applications-topics-extraction-with-nmf-lda-py
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([ feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1] ]))
    print()

def save_top_words(model, feature_names, n_top_words):
    wordme = {}
    for topic_idx, topic in tqdm(enumerate(model.components_)):
        wordme["Topic #%d" % topic_idx] = [ feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1] ]
    return wordme

def show_keywords(centroids, terms, n_top_words):
    n_clusters = centroids.shape[0]
    keyword_clusters = {}
    for i in tqdm(range(n_clusters)):
        # keyword_clusters['Topic #%d' % i] = [ terms[k] for k in centroids.argsort()[:,::-1][i, :n_top_words] ]
        keyword_clusters['Topic #%d' % i] = [ terms[k] for k in centroids.argsort()[:-n_top_words - 1:-1]]
    return keyword_clusters

# Collapse documents per dataset to get better idf
def create_doc(entry):
    doc = entry[u'description'] + ' ' + entry[u'title']
    for keyword in entry[u'keyword']:
        doc = doc + ' ' + keyword
    return doc

if __name__ == "__main__":
    #####################################################################################################################
    # Setting Options
    #####################################################################################################################
    # Number of features to keep after weighting
    prop_feat = 1000
    # Number of components to keep after dimension reduction
    prop_comp = 100
    # Number of clusters to generate
    prop_clust = 60
    # Words to keep
    prop_words = 30
    # Stopwords to exclude
    add_stopwords = ['department', 'of', 'commerce', 'doc', 'noaa', 'national', 'data', 'and', 'the', 'for', 'centers', 'united', 'states']
    stopwords = ENGLISH_STOP_WORDS.union(add_stopwords)
    print '#####################################################################################################################'
    print 'Options selected:'
    print '#####################################################################################################################'
    print '# of features retained: %s, # of components: %s, # of clusters: %s, # of tags: %s'%(prop_feat, prop_comp, prop_clust, prop_words)
    print '#####################################################################################################################'
    print 'Stopwords used:'
    print '#####################################################################################################################'
    pp.pprint(stopwords)
    #####################################################################################################################
    # Open file
    #####################################################################################################################
    print '#####################################################################################################################'
    print 'Generating X input'
    print '#####################################################################################################################'
    with open('./noaa.json', 'rb') as f:
        noaa = json.load(f)
    X_pre = []
    for entry in tqdm(noaa):
        X_pre.append(create_doc(entry))
    #####################################################################################################################
    # Processing ngrams to weight
    #####################################################################################################################
    print '#####################################################################################################################'
    print 'Weighting ngrams'
    print '#####################################################################################################################'
    t0 = time.time()
    print 'Time started: %s'%t0
    vect = TfidfVectorizer(ngram_range=(1,3), stop_words=stopwords, max_features=prop_feat)
    # vect = CountVectorizer(ngram_range=(1,1), stop_words=stopwords, max_features=prop_feat)
    X_vect = vect.fit_transform(X_pre)
    print 'Time elapsed: %s'%(time.time()-t0)
    print 'n_samples: %s, n_features: %s'%X_vect.shape
    with open('feature_list.json', 'wb') as f:
        json.dump(vect.get_feature_names(), f)
    #####################################################################################################################
    # Dimesionality reduction on sparse matrix
    #####################################################################################################################
    print '#####################################################################################################################'
    print 'Dimensionality reduction'
    print '#####################################################################################################################'
    t0 = time.time()
    print 'Time started: %s'%t0
    dim = TruncatedSVD(n_components=prop_comp)
    X_dim = dim.fit_transform(X_vect)
    print 'LSA Time elapsed: %s'%(time.time()-t0)

    # pp.pprint(dim.components_)
    #####################################################################################################################
    # L2 distance normalizer for K-means clustering
    #####################################################################################################################
    print '#####################################################################################################################'
    print 'Nomalizing'
    print '#####################################################################################################################'
    t0 = time.time()
    print 'Time started: %s'%t0
    norm = Normalizer()
    X_norm = norm.fit_transform(X_dim)
    print 'Time elapsed: %s'%(time.time()-t0)
    #####################################################################################################################
    # Topic Generation
    #####################################################################################################################
    # print '#####################################################################################################################'
    # print 'Topic Generation: KMeans'
    # print '#####################################################################################################################'
    # t0 = time.time()
    # print 'Time started: %s'%t0
    # clust = KMeans(n_clusters=prop_clust, max_iter=100)
    # X_clust = clust.fit_transform(X_norm)
    # print 'Time elapsed: %s'%(time.time()-t0)
    # centroids = dim.inverse_transform(clust.cluster_centers_)
    # with open('topic_lsa_kmeans.json', 'wb') as f:
    #     json.dump(show_keywords(centroids, vect.get_feature_names(), prop_words), f)
    # print '#####################################################################################################################'
    # print 'Topic Generation: Latent Dirichlet Allocation'
    # print '#####################################################################################################################'
    # t0 = time.time()
    # print 'Time started: %s'%t0
    # lda = LatentDirichletAllocation(n_topics=prop_clust, max_iter=5).fit(X_vect)
    # print 'LDA Time elapsed: %s'%(time.time()-t0)
    # with open('topic_lda.json', 'wb') as f:
    #     json.dump(save_top_words(lda, vect.get_feature_names(), prop_words), f)
    print '#####################################################################################################################'
    print 'Topic Generation: Dirichlet Process Gaussian Mixture Model'
    print '#####################################################################################################################'
    t0 = time.time()
    print 'Time started: %s'%t0
    dpgmm = DPGMM(n_components=prop_clust, alpha=100).fit(X_norm)
    pp.pprint(dpgmm.means_)
    # pp.pprint(dpgmm.predict(X_norm))
    print 'DPGMM Time elapsed: %s'%(time.time()-t0)
    # with open('topic_dpgmm.json', 'wb') as f:
    #     json.dump(save_top_words(lda, vect.get_feature_names(), prop_words), f)




