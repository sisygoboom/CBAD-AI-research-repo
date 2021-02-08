from StandardiseDatasets import StandardiseDatasets
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

for i in ['https', 't', 'm', 'co', 'rt', 's', 're', 'go', 'use', 'y', 'feel', 
          'name', 'll', 'another', 'via', 'da', 'said', 'user', 'u', 'say', 
          'got', 'see', 'know', 'im', 'lol', 'try', 'look', 'want', 'never',
          'even', 'need', 'still', 'amp', 'us', 'really', 'one', 'real', 'will',
          'time', 'day', 'alway', 'Van', 'looks', 'word', 'back', 'yo', 'ya',
          'done', 'win', 'new', 'man', 'think', 'give', 'life', 'make', 'ain',
          'Happy', 'don', 'let', 'tell', 'good', 'stop', 'call', 'people', 'now',
          'card', 'bout', 'going', 'every', 'come', 'Full', "ain't", 'right',
          'Oh', '0h', 'year', 'bad', 'gonna', 'called', 'wanna', 'put', 'today',
          ]:
    STOPWORDS.add(i)

sd = StandardiseDatasets()

data = sd.get_all()

x = data['Tweet']

x = ' '.join(x)
x.lower()

# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");
    
# Generate word cloud
wordcloud = WordCloud(width = 3000, height = 2000, max_words=30000, random_state=1, background_color='white', colormap='Dark2_r', collocations=False, stopwords = STOPWORDS).generate(x)
# Plot
plot_cloud(wordcloud)