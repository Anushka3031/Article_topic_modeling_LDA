from flask import Flask, render_template, request
import joblib
import gensim
from gensim.models.phrases import Phrases, Phraser
from gensim.parsing.preprocessing import remove_stopwords
from gensim.corpora import Dictionary
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('Article_Topic_Modeling.h5')
# Define the topic names
topic_names = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        title = request.form['title']
        abstract = request.form['abstract']
        
        # Preprocess the input text (replace with your preprocessing code)
        article = (title + ' ' + abstract)  
        article_words= list(sent_to_words(article))
        new_article = [preprocess_text(text) for text in article_words] 

        
        #create dictionary
        dictionary = Dictionary(new_article)
  
        new_corpus = [dictionary.doc2bow(doc) for doc in new_article]
        # Make the prediction using the pre-trained model
        topic = model.get_document_topics(new_corpus)

        for i, article in enumerate(article):
            # Sort the topics by score in descending order
            sorted_topics = sorted(topic[i], key=lambda x: x[1], reverse=True)
    
            # Get the top one or two topic names
            num_topics_to_display = min(2, len(sorted_topics))
            predicted_topic_names = [topic_names[topic[0]] for topic in sorted_topics[:num_topics_to_display]]
    
            # Print the predicted topic names for the article
            #print(f"Article ID #{article['ID']} Predicted Topics: {', '.join(predicted_topic_names)}")
        
            # Render the result template with the predicted topic name
            return render_template('result.html', title=title, abstract=abstract,topic=predicted_topic_names)
    
    # Render the main template
    return render_template('index.html')

def preprocess_text(text):
        custom_filters = [
        lambda x: x.lower(),          # lowercase
        lambda x: x.strip(),          # Remove whitespaces
        remove_stopwords,             # Remove stop words
    ]

        processed_texts = []
        for item in text:
            processed_text = item
            for filter_func in custom_filters:
                processed_text = filter_func(processed_text)
                processed_texts.append(processed_text)  # Append the processed tokens as a list
    
    # Generate bigrams and trigrams
        bigram = Phrases(processed_texts, min_count=5, threshold=100)
        trigram = Phrases(bigram[processed_texts], min_count=5, threshold=100)

    # Apply bigrams and trigrams to prcessed texts
        processed_texts = trigram[bigram[processed_texts]]
        return processed_texts

def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)

if __name__ == '__main__':
    app.run(debug=True)