## E-commerce Recommendation System using SVD
## 📌 Problem Statement
In modern e-commerce platforms, customers interact with items through actions such as views, add-to-cart, and transactions. However, most users only interact with a fraction of available items, making it difficult for them to discover products they are likely to purchase.
Without an effective recommendation system, businesses risk:
- Low product discoverability
- Reduced user engagement
- Missed sales opportunities

This project addresses this challenge by building a personalized recommendation engine using Singular Value Decomposition (SVD) on implicit feedback data.


![REcommendation-System](https://github.com/ioakowuah/Recommendation-System-Analysis-and-Modeling/blob/main/recommendation%20system%20.png)

##  Project Details

- **Project Title:** Recommendation-System-Analysis-and-Modeling 
- **Tool Used:** Python  
- **Dataset Source:** [E-Commerce Dataset](https://huggingface.co/datasets/ioakowuah/RecommendationSystem)  
- **Period Covered:** January 2015 – December 2015  


## Exploratory Data Analysis (EDA)
Before building the recommendation engine, I conducted EDA to better understand the dataset:
#### Key Insights
-	User activity:
* View Events Dominate: The majority of user interactions are views, with over 16 million occurrences. This suggests that users frequently browse items but do not necessarily proceed

* Low Add-to-Cart Conversion: The number of add-to-cart events is significantly lower than views. This indicates a high drop-off rate between viewing an item and adding it to the cart.

- Minimal Transactions: Purchases (transactions) are the least frequent event type. This suggests that only a small fraction of users complete the buying process.


-	Item popularity:
- Item (ID: 37029) has 24,472 count of views but 570 count of purchases. This indicate this item has highest potential of adding to revenue should we prioritize it in marketing and sales campaigns.
- The other top 4 items high view counts without appearance in the top 5 item purchases indicate an opportunity to scale up conversions.

-	Session patterns:
- More purchases, 42.3% are made in the midnight between the hours of 9pm to 5 am.

![Key Insight](https://github.com/ioakowuah/Recommendation-System-Analysis-and-Modeling/blob/main/What%20events%20are%20exhibited%20by%20users%20on%20the%20ecommerce%20website.png)

![Key Insight](https://github.com/ioakowuah/Recommendation-System-Analysis-and-Modeling/blob/main/top%205%20viewed%20and%20transacted.png)

![Key Insight](https://github.com/ioakowuah/Recommendation-System-Analysis-and-Modeling/blob/main/purchased%20over%20the%20periods%20of%20the%20day.png)

