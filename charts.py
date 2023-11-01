import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# Define the categories and their probabilities
categories = ['OPENESS', 'CONSCIENTIOUSNESS', 'EXTRAVERSION', 'AGREEABLENESS', 'NEUROTICISM']

def create_charts(predictions, text, uid):
    # Personality predictions
    personality = pd.DataFrame(list(zip(categories, predictions["score"], predictions["prob"])),
                   columns =['Category', 'Score', "Probability"])

    # 'Probability of Trait'
    fig_1, ax_1 = plt.subplots(figsize=(18, 12))
    barchart_1= sns.barplot(data=personality.sort_values(by=['Probability'], ascending=False), x="Category", y="Probability");
    barchart_1.bar_label(ax_1.containers[0], label_type='edge', padding=15);
    plt.savefig(f"static/image/{uid}_traitprob.png")

    # 'Trait Score'
    fig_2, ax_2 = plt.subplots(figsize=(18,12))
    barchart_2 = sns.barplot(data=personality.sort_values(by=['Probability'], ascending=False), x="Category", y="Score");
    barchart_2.bar_label(ax_2.containers[0], label_type='edge', padding=15);
    plt.savefig(f"static/image/{uid}_traitscore.png")


    specialisation = []
    words = []
    count = []
    for key, item in predictions["key_word"].items():
        specialisation.append(key)
        words.append(item[1])
        count.append(item[0])

    key_words = pd.DataFrame(list(zip(specialisation, words, count)),
                   columns =['Domain', 'Words', "Count"])

    # 'Word extraction'
    fig_3, ax_3 = plt.subplots(figsize=(18,12))
    barchart_3 = sns.barplot(data=key_words.sort_values(by=['Count'], ascending=False), x="Domain", y="Count");
    barchart_3.bar_label(ax_3.containers[0], label_type='edge', padding=15);
    plt.savefig(f"static/image/{uid}_keyword.png")

    ## Wordcloud creation
    cloud = WordCloud().generate(text)
    cloud.to_file(f"static/image/{uid}_wordcloud.png")
