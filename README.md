## E-commerce Recommendation System using SVD
## 📌 Problem Statement
In modern e-commerce platforms, customers interact with items through actions such as views, add-to-cart, and transactions. However, most users only interact with a fraction of available items, making it difficult for them to discover products they are likely to purchase.
Without an effective recommendation system, businesses risk:
- Low product discoverability
- Reduced user engagement
- Missed sales opportunities

This project addresses this challenge by building a personalized recommendation engine using Singular Value Decomposition (SVD) on implicit feedback data.


![REcommendation-System](https://github.com/ioakowuah/Recommendation-System-Analysis-and-Modeling/blob/main/recommendation%20system%20.png)


##  Link to Deployed Recommendation System Demo
- **Website:** [Link](https://recommendation-system-analysis-and-modeling-braonjkwfzfbdyxi9f.streamlit.app/)
  
##  Project Details

- **Project Title:** Recommendation-System-Analysis-and-Modeling 
- **Tool Used:** Python  
- **Dataset Source:** [E-Commerce Dataset](https://huggingface.co/datasets/ioakowuah/RecommendationSystem)  
- **Period Covered:** January 2015 – December 2015  


## Exploratory Data Analysis (EDA)
Before building the recommendation engine, I conducted EDA to better understand the dataset:
#### Key Insights
####	User activity:
* View Events Dominate: The majority of user interactions are views (96.7%). This suggests that users frequently browse items but do not necessarily proceed

* Low Add-to-Cart Conversion: The number of add-to-cart events(2.5%) is significantly lower than views. This indicates a high drop-off rate between viewing an item and adding it to the cart.

- Minimal Transactions: Purchases (transactions - 0.8%) are the least frequent event type. This suggests that only a small fraction of users complete the buying process.


####	Item popularity:
- Item (ID: 37029) has 24,472 count of views but 570 count of purchases. This indicate this item has highest potential of adding to revenue should we prioritize it in marketing and sales campaigns.
- The other top 4 items high view counts without appearance in the top 5 item purchases indicate an opportunity to scale up conversions.

####	Session patterns:
- More purchases, 42.3% are made in the midnight between the hours of 9pm to 5 am.

![Key Insight](https://github.com/ioakowuah/Recommendation-System-Analysis-and-Modeling/blob/main/What%20events%20are%20exhibited%20by%20users%20on%20the%20ecommerce%20website(percentage).png)

![Key Insight](https://github.com/ioakowuah/Recommendation-System-Analysis-and-Modeling/blob/main/5%20viewed%20and%20transacted%20top%20items.png)

![Key Insight](https://github.com/ioakowuah/Recommendation-System-Analysis-and-Modeling/blob/main/purchased%20over%20the%20periods%20of%20the%20day.png)

#### 🚀Solution
We developed a matrix factorization-based recommender system that learns user preferences and item representations from historical event logs.
-	Input: User-item interactions (view, addtocart, transaction)
-	Transformation: Assign implicit ratings (view=1, addtocart=3, transaction=5)
-	Model: TruncatedSVD (dimensionality reduction on sparse user–item matrix)
-	Output: Top-N personalized product recommendations for each user
Additionally, we provide evaluation metrics (Precision@K) to measure recommendation quality.

####  🛠️Tech Stack
-	Python 🐍
-	Pandas / NumPy – data preprocessing
-	Scipy (sparse matrices) – efficient storage of large user-item interactions
-	Scikit-learn (TruncatedSVD) – matrix factorization
-	Pickle / JSON – model and mapping persistence
-	Streamlit (optional) – interactive recommendation interface

#### 📈 Impact
- Improves user experience by surfacing relevant products
- Increases engagement and time spent on platform
- Boosts conversion rate & revenue by recommending products users are likely to buy
- Provides scalable offline model training with efficient sparse matrix factorization


