import re, json, pprint
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from tqdm import *

pp = pprint.PrettyPrinter(indent=2)

def show_keywords(datajson, centroids, terms):
    n_clusters = centroids.shape[0]
    keyword_clusters = []
    for i in range(n_clusters):
        keyword_clusters.append([ terms[k] for k in centroids.argsort()[:,::-1][i, :20] ])
        for ind in centroids.argsort()[:,::-1][i, :20]:
            print ' %s' % terms[ind]
        print
    return keyword_clusters

if __name__ == "__main__":
    print 'Preprocessing'
    with open('./noaa.json', 'rb') as f:
        noaa = json.load(f)
    X_pre = []
    for entry in tqdm(noaa):
        X_pre.append(entry[u'description'])
        X_pre.extend(entry[u'keyword'])
    stopwords = ['department', 'of', 'commerce', 'doc', 'noaa', 'national', 'data', 'and', 'the', 'for', 'centers', 'united', 'states']
    tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words=stopwords)
    X_tfidf = tfidf.fit_transform(X_pre)
    print 'Output feature name list'
    with open('feature_names.json', 'wb') as f:
        json.dump(tfidf.get_feature_names(), f)
    print 'Dimensionality Reduction'
    lsa = TruncatedSVD(n_components=5)
    X_lsa = lsa.fit_transform(X_tfidf)
    norm = Normalizer()
    X_norm = norm.fit_transform(X_lsa)
    print 'Clustering'
    kmeans = KMeans(n_clusters=30, max_iter=100)
    kmeans.fit(X_norm)
    centroids = lsa.inverse_transform(kmeans.cluster_centers_)
    print 'Get Keywords'
    with open('examine_keys.json', 'wb') as f:
        json.dump(show_keywords(noaa, centroids, tfidf.get_feature_names()), f)

