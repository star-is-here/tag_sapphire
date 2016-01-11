import re, json, pprint
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from tqdm import *

pp = pprint.PrettyPrinter(indent=2)

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
    # Open file
    with open('./noaa.json', 'rb') as f:
        noaa = json.load(f)
    X_pre = []
    for entry in tqdm(noaa):
        # X_pre.append(entry[u'description'])
        # X_pre.extend(entry[u'keyword'])
        X_pre.append(create_doc(entry))
    stopwords = ['department', 'of', 'commerce', 'doc', 'noaa', 'national', 'data', 'and', 'the', 'for', 'centers', 'united', 'states']
    tfidf = TfidfVectorizer(ngram_range=(1,3), stop_words=stopwords).fit(X_pre)
    prop_comp = round(tfidf.idf_.shape[0] * .01, 0)
    prop_clust = round(prop_comp * 0.01, 0)
    # Dump for once over
    with open('feature_names.json', 'wb') as f:
        json.dump(tfidf.get_feature_names(), f)
    lsa = TruncatedSVD(n_components=prop_comp)
    norm = Normalizer()
    kmeans = KMeans(n_clusters=prop_clust, max_iter=100)
    pipeline = make_pipeline(tfidf, lsa, norm, kmeans)
    pipeline.fit_transform(X_pre)
    centroids = lsa.inverse_transform(kmeans.cluster_centers_)
    with open('examine_keys.json', 'wb') as f:
        json.dump(show_keywords(centroids, tfidf.get_feature_names()), f)
    # X_test = [ noaa[0][u'description'] ]
    # X_test.extend(noaa[0][u'keyword'])
    # print len(X_test)
    # cluster_test = pipeline.predict(X_test)
    # pp.pprint(cluster_test)
    # print cluster_test.shape
    # Predict
    # Build set
    # X_hat = { entry[u'title']: }



