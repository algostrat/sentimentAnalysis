from textblob import TextBlob

opinion = TextBlob("Finished product. Let's go!!!!\n\n#playstation5 #console  #showmeyours #begreater #beyourself  #PlayStation\n#sony https://t.co/TcEGW1HsWz")

print(opinion.sentiment)


opinion = TextBlob("sup dawg, how you been?")
opinion = TextBlob("I'm greatly #disappointed")