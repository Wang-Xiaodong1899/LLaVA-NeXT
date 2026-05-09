from sklearn.feature_extraction.text import CountVectorizer

sentence = """The game being played in the video is dodgeball, 
              a team sport typically played in a large indoor or outdoor court. 
              The objective of the game is to eliminate all the players on the opposing team 
              by hitting them with a ball while they are not behind a designated wall 
              or standing within a designated area called the \"safe zone.\""""

vectorizer = CountVectorizer()


X = vectorizer.fit_transform([sentence])


print(vectorizer.get_feature_names_out())
print(X.toarray())
