from sentence_transformers import SentenceTransformer, util
sentences = ['Umbrella found which is like a newspaper',
              'Pasta found',
             'Found phone samsung',
             'Found laptop',
             'Found TV',
             'Found realme phone',
             'Found gautham near prateeksha']
model = SentenceTransformer('all-MiniLM-L6-v2')
def search(inputtext,database):
    sentences=inputtext+database
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    pairs = []
    for i in range(len(cosine_scores)-1):
        for j in range(i+1, len(cosine_scores)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    for pair in pairs[0:10]:
        i, j = pair['index']
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))
    return pairs
print(search(['Found an Umbrella like a newspaper'],sentences))