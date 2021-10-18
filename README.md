# Sentiment Analysis project for FAU's Engineering Design course, group 24
For this project, we gathered social media posts data from platforms such as Twitter, and determined which posts are trending, abusive, and their overall sentiment. After a post had been retrieved and analyzed by our backend components (or systems), we showed the post (and metadata about it, such as when it was posted, by whom, etc) and our analysis of it on our front-end web-based platform. To develop this application, we used several different technologies. To ensure that our application could be easily scalable, we placed an emphasis on modular design and developed multiple components (or modules) to perform various tasks. For example, we have developed a social media crawling component that is responsible for gathering the latest social media post data, a sentiment analysis component, an abusive language analysis component, a trending analysis component, and front-end component(s). These components interact with each other to transform, massage, and ultimately, present data in a simple, interactive, and informative manner. The solution we are developing will be beneficial to users of all fields. The data we present contains information that could aid those in law enforcement, marketing, news organizations, and more.

This proposal responds to an RFP from rewardStyle Social Media / Blog – Sentiment Analysis. RewardStyle seeks a solution to the problem of analyzing and predicting sentiment on social media posts. The company rewardStyle requests a web application dashboard that displays sentiment analysis, trend discovery, and abuse flagging data.

Sentiment Analysis can help rewardStyle better understand public reaction to a specific product. Having the ability to observe sentiment, discover trends and see abusive posts can significantly improve rewardStyles future decision making. The flagging of abusive posts is also an important feature that allows for a safer internet experience. In general, sentiment analysis can be used by rewardStyle and many other companies to gain insight on social media posts. 

The goal for this project is to acquire specific data from Twitter and research the feasibility for scraping Twitter data. The specific data will be in accordance with what Kurt Radwanski (rewardStyle engineer) suggested, which is posts involving holiday themed hashtags. The data will be used to analyze posts for sentiment scores that will then be plotted. As of this proposal, a lite version of a crawler for downloading twitter data has been established but a plan to automate this server to download data on a timely basis will be completed. We will demonstrate the front-end application and preliminary discoveries and possible algorithms as to how to analyze trends and further classify social media blog posts to an end user.

### Git Repo file structure



## Overall System Design with Block Diagrams
Below is a visual representation of the system in the form of a block diagram. It is important to note how each component is interacting with each other. (See 2.4 for high level explanation and section 3 for implementation)

## Sub-System Design and High-Level Implementation:
Data Preprocessing Component
The data preprocessing component requires a lot of testing to ensure its reliability in providing clean data to the subsequent algorithms. Other than looking at how text blog posts are transformed through this component it was very hard to analyze performance from a high level. Rather, we did the best to see how this component performs by looking at cases of faulty classification in the sentiment analysis component. We noted that in our preliminary implementation many blog posts instances were classified incorrectly by python’s Textblob. We mentioned in the implementation how this component ensures clean data to the subsequent algorithms and testing of the component was done using a SCRUM process of checking how the inclusion and modification of different textual preprocessors affect the scoring of the sentiment analysis algorithm. 


### cSentiment Analysis Component
Testing for the sentiment was done by analyzing classification performances with known labeled datasets such as the Analytics Vidhya tweet sentiment datasets (Twitter Sentiment Analysis), and the Sentiment140 dataset from Kaggle (Sentiment140). Since we used an off-the-shelf sentiment algorithm from TextBlog and/or Vader we compared on a high level how plots of sentiment scores compare for different algorithms on known and unknown data instances. This high-level overview is in accordance with the user requirements of viewing overall trends in trends and not necessarily relying on sentiment to produce intuitive information to the user. Since, our sentiment analysis algorithm did not produce a large number of false negatives, positives, or instances where no classification was provided, it ensures that this component is working correctly. 

### Trend Discovery Component
Trend discovery was tested by comparing our discovered trending topics with word cloud performance from python’s wordcloud library where we compared the occurrences of just the nouns. If the word cloud generated similar results, we knew that word frequencies are being accounted for. It was extremely important to test new model versions across the same flow of data given that testing/training in a bubble or across different data sets to see the performance those new models yield compared to earlier versions overall, we saw improvement in accuracy. This accuracy was checked through use of a dataset for which we manually found the trends for. That is, partitioning a separate data subset, exporting the text blog posts into a visual format, and manually going through every post to see if common occurring trends that we saw were being detected by the trend discovery algorithm component. 

### Abuse Discovery Component
The abuse discovery algorithm was built with a training set but a sizable portion was set aside as a testing set. If we regard the training and testing dataset to be highly accurate, which they are because they were used in actual Kaggle competitions, then we got the actual performance metrics for this component. 


### Web Application Component
To test the web application component of this project we followed the scrum methodology. We performed unit testing and functionality testing in order to determine problems within our application. Unit testing allowed us to test individual functions of the code. For example, it made sure that text is displayed properly.  While Functional testing allowed us to test the functionality of the actual web application. For example, testing the widgets and buttons. 

There are some tests that we performed that are testing the callbacks and the original posts. Black box testing is another form of testing that allowed us to make sure that the proper functions of the application are being executed. Testing specific portions of the web application will ensure that they are performing correctly. For example, we clicked on specific posts from the chart and made sure that it will take the user back to the original social media post. We also tested the dashboard buttons to make sure that they perform as they should. Lastly, we debugged the program to ensure a high quality of code. 

### Crawler Component
The crawler features a modular design; each module of the crawler was independently tested. The modules in the current implementation consist of the data access module, crawler coordination (main) module, social media data source module(s), and REST helper module. The data access module can be tested by confirming database connection, by providing it with dummy data for storage insertion, and by retrieving and displaying the raw results after a query. The crawler coordination module oversees acquiring data from the data source modules and supplying it to the storage module. This module was tested by disabling the storage module and displaying the results returned by the data source modules to ensure data is being retrieved, and then by disabling the pipeline from the data source(s) module and supplying the storage module with dummy data. The social media data source modules were tested independently by dumping raw results of the responses from the API endpoints, to ensure that data is being retrieved appropriately. The REST helper module was responsible for coordinating the endpoint connections (on a socket and HTTP layer) and was tested by sending requests and showing responses to ensure connections to the endpoints are made.


![blockdiagram](https://user-images.githubusercontent.com/8900863/137652503-a7fbd46e-3b09-45f2-be86-034e04251fa9.JPG)

![statediagram](https://user-images.githubusercontent.com/8900863/137652511-4a28e19e-5642-448a-a961-aabb0758d0ec.JPG)

![usecasediagram](https://user-images.githubusercontent.com/8900863/137652517-fcf117b4-0d30-4cd9-8eac-3d09a399024f.JPG)



