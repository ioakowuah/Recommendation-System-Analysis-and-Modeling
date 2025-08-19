# Recommendation-System-Analysis-and-Modeling
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
-	Event distribution:
o ~70% of events were views |
o	~20% were add-to-cart |
o	~10% were transactions
-	User activity:
o	A small fraction of users contribute to the majority of events (typical power-law distribution). |
o	Some users are very active, while many only appear once or twice. |
-	Item popularity:
o	A few items are extremely popular (highly viewed/purchased). |
o	Long-tail effect: majority of items have very few interactions. |
-	Session patterns:
o	Users who added items to the cart were significantly more likely to purchase within the same session.

