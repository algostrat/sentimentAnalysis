# sentimentAnalysis
For this project, we gathered social media posts data from platforms such as Twitter, and determined which posts are trending, abusive, and their overall sentiment. After a post had been retrieved and analyzed by our backend components (or systems), we showed the post (and metadata about it, such as when it was posted, by whom, etc) and our analysis of it on our front-end web-based platform. To develop this application, we used several different technologies. To ensure that our application could be easily scalable, we placed an emphasis on modular design and developed multiple components (or modules) to perform various tasks. For example, we have developed a social media crawling component that is responsible for gathering the latest social media post data, a sentiment analysis component, an abusive language analysis component, a trending analysis component, and front-end component(s). These components interact with each other to transform, massage, and ultimately, present data in a simple, interactive, and informative manner. The solution we are developing will be beneficial to users of all fields. The data we present contains information that could aid those in law enforcement, marketing, news organizations, and more.

This proposal responds to an RFP from rewardStyle Social Media / Blog – Sentiment Analysis. RewardStyle seeks a solution to the problem of analyzing and predicting sentiment on social media posts. The company rewardStyle requests a web application dashboard that displays sentiment analysis, trend discovery, and abuse flagging data.

Sentiment Analysis can help rewardStyle better understand public reaction to a specific product. Having the ability to observe sentiment, discover trends and see abusive posts can significantly improve rewardStyles future decision making. The flagging of abusive posts is also an important feature that allows for a safer internet experience. In general, sentiment analysis can be used by rewardStyle and many other companies to gain insight on social media posts. 

The goal for this project is to acquire specific data from Twitter and research the feasibility for scraping Twitter data. The specific data will be in accordance with what Kurt Radwanski (rewardStyle engineer) suggested, which is posts involving holiday themed hashtags. The data will be used to analyze posts for sentiment scores that will then be plotted. As of this proposal, a lite version of a crawler for downloading twitter data has been established but a plan to automate this server to download data on a timely basis will be completed. We will demonstrate the front-end application and preliminary discoveries and possible algorithms as to how to analyze trends and further classify social media blog posts to an end user.

Below is a visual representation of the system in the form of a block diagram. It is important to note how each component is interacting with each other. (See 2.4 for high level explanation and section 3 for implementation)

![block diagram](readmeimages/blockdiagram.png)
