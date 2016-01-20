import re, json, pprint, time
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from tqdm import *

pp = pprint.PrettyPrinter(indent=2)

# taken from sklearn example http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf_lda.html#example-applications-topics-extraction-with-nmf-lda-py
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def save_top_words(model, feature_names, n_top_words):
    wordme = {}
    for topic_idx, topic in tqdm(enumerate(model.components_)):
        wordme["Topic #%d:" % topic_idx] = [ feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1] ]
    return wordme

def show_keywords(centroids, terms):
    n_clusters = centroids.shape[0]
    keyword_clusters = []
    for i in range(n_clusters):
        keyword_clusters.append([ terms[k] for k in centroids.argsort()[:,::-1][i, :20] ])
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
    prop_feat = 20000
    # Number of components to keep after dimension reduction
    prop_comp = 200
    # Number of clusters to generate
    prop_clust = 200 
    # Stopwords to exclude
    stopwords = ['department', 'of', 'commerce', 'doc', 'noaa', 'national', 'data', 'and', 'the', 'for', 'centers', 'united', 'states']
    print '#####################################################################################################################'
    print 'Options selected:'
    print '#####################################################################################################################'
    print '# of features retained: %3d, # of components: %3d, # of clusters: %3d'%(int(prop_feat), int(prop_comp), int(prop_clust))
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
        # X_pre.append(entry[u'description'])
        # X_pre.extend(entry[u'keyword'])
        X_pre.append(create_doc(entry))
    #####################################################################################################################
    # Processing ngrams to weight
    #####################################################################################################################
    print '#####################################################################################################################'
    print 'Weighting ngrams'
    print '#####################################################################################################################'
    t0 = time.time()
    vect = TfidfVectorizer(ngram_range=(1,3), stop_words=stopwords, max_features=prop_feat)
    # vect = CountVectorizer(ngram_range=(1,1), stop_words=stopwords, max_features=prop_feat)
    X_vect = vect.fit_transform(X_pre)
    print 'Time elapsed: %s'%(time.time()-t0)
    pp.pprint(X_vect.shape)
    with open('feature_list.json', 'wb') as f:
        json.dump(vect.get_feature_names(), f)
    #####################################################################################################################
    # Dimesionality reduction on sparse matrix
    #####################################################################################################################
    print '#####################################################################################################################'
    print 'Dimensionality reduction'
    print '#####################################################################################################################'
    t0 = time.time()
    # dimred = TruncatedSVD(n_components=prop_comp, n_jobs=-1)
    dimred = LatentDirichletAllocation(n_topics=prop_comp, max_iter=5, n_jobs=1)
    dimred.fit(X_vect)
    print 'Time elapsed: %s'%(time.time()-t0)
    # pp.pprint(dimred.components_)
    # print_top_words(dimred, vect.get_feature_names(), 30)
    with open('dimensions_kept.json', 'wb') as f:
        json.dump(save_top_words(dimred, vect.get_feature_names(), 30), f)
    #####################################################################################################################
    # L2 distance normalizer for K-means clustering
    #####################################################################################################################
    # print '#####################################################################################################################'
    # print 'Nomalizing'
    # print '#####################################################################################################################'
    # norm = Normalizer()
    #####################################################################################################################
    # K-means clustering on reduced dimensions
    #####################################################################################################################
    # print '#####################################################################################################################'
    # print 'Clustering'
    # print '#####################################################################################################################'
    # kmeans = KMeans(n_clusters=prop_clust, max_iter=100)
    # # pipeline_lsa = make_pipeline(tfidf, lsa, norm, kmeans)
    # pipeline_lda = make_pipeline(cntvect, lda, norm, kmeans)
    # pipeline_lda.fit_transform(X_pre)
    # centroids = lsa.inverse_transform(kmeans.cluster_centers_)
    # with open('examine_keys.json', 'wb') as f:
    #     json.dump(show_keywords(centroids, cntvect.get_feature_names()), f)
    # X_test = [ noaa[0][u'description'] ]
    # X_test.extend(noaa[0][u'keyword'])
    # print len(X_test)
    # cluster_test = pipeline.predict(X_test)
    # pp.pprint(cluster_test)
    # print cluster_test.shape
    # Predict
    # Build set
    # X_hat = { entry[u'title']: }



