# Technical Report

**Project: Toxicity of Reddit Comments in Singapore**  
**Members: And Lin Xuan, Chan Rui Qi Stacy, Lim Shu Yau, Tan Xuan Huan, Tong Zhixian**  
Last updated on 13/11/2024

## Section 1: Context
This project originated from growing concerns within the Ministry of Digital Development and Information’s (MDDI) Online Trust and Safety department regarding the rising levels of toxicity and hate speech on social media platforms. Over the past few years, there has been increasing evidence that online spaces are fostering more polarising and extreme viewpoints, with individuals expressing hate or toxicity in their comments and posts. An Online Safety Poll conducted by MDDI highlighted the urgency of this issue, with findings that 66% of survey respondents encountered harmful content on social media this year, up from 57% in the previous year. Recognising the need to better understand and address this trend, MDDI tasked us with this project to conduct a comprehensive, data-driven analysis using Reddit data from popular Singapore subreddits to evaluate whether discussions have become increasingly hateful and toxic in recent years and to identify possible reasons behind these observed trends.

## Section 2: Scope
### 2.1 Problem
The key problem addressed by this project is the rising level of toxicity and hate speech on social media platforms in the past few years, a pressing concern for the Online Trust and Safety department at MDDI, which oversees online issues related to misinformation, deep fakes, and toxic content.
The significance of this problem is substantial. Left unaddressed, online toxicity can deepen societal polarisation and strain social cohesion, a particularly sensitive issue in Singapore’s diverse, multi-racial, and multi-religious context. Additionally, the increased exposure of youth to harmful and toxic content raises further concerns about their mental health and social well-being.

Data science and machine learning provide essential tools for solving this problem by enabling large-scale, nuanced analysis of toxicity trends and patterns, which is impossible to do manually. Natural Language Processing (NLP) techniques are particularly effective for analysing text data, as they allow for the identification, quantification, and tracking of toxic content over time with greater efficiency and accuracy than manual methods. Additionally, topic modelling allows us to uncover patterns and possible triggers of toxicity, delivering actionable insights for MDDI and relevant stakeholders.

### 2.2 Success Criteria
To assess the success for this project, we focus on three primary criteria that align closely with MDDI policy team’s key objectives: 

#### 2.2.1 Demonstration of toxicity trends
Success will first be measured by our ability to effectively illustrate the evolution of toxicity within Singapore-specific Reddit communities over time. This includes generating visualisations and conducting analysis that highlight shifts or patterns in toxic discourse, providing the policy team with a clear view of the problem’s magnitude. The goal is to achieve a clear and statistically significant representation of how toxicity has escalated or declined, ensuring actionable insights into temporal trends that may correlate with external factors, such as major events or policy changes. 

#### 2.2.2 Identification of key drivers of toxicity
Another measure of success will be the identification of specific topics or factors that substantially contribute to rising toxicity levels. By utilising techniques such as topic modelling, the project will pinpoint at least two major drivers of toxicity within these communities. These insights are intended to equip the policy team with an understanding of the underlying causes, allowing them to develop targeted strategies for toxicity reduction. Success in this criterion will be achieved if the analysis highlights distinct topics or recurrent themes that significantly correlate with increased toxic behaviour, as validated by statistical metrics.

#### 2.2.3 Provision of actionable insights and recommendations
The final measure of success will involve delivering clear recommendations for toxicity mitigation based on our findings. These recommendations should outline concrete steps that the policy team can take. Success will be measured by the practicality and relevance of these recommendations to the policy stakeholders. 

### 2.3 Assumptions
For our analysis and findings to be valid, several key assumptions have been made in this project:

#### 2.3.1 Subreddit representativeness 
This project assumes that the selected Singapore-specific subreddits reflect broader sentiments and discussions within Singapore’s social media landscape. If this is not the case, the insights derived may be limited in their ability to inform wider social or policy recommendations regarding toxicity in social media.

#### 2.3.2 Consistent subreddit activity and demographics
We assume that the activity levels and demographic composition of the three subreddit communities analysed will remain relatively consistent over the given period. Significant shifts in user engagement such as in the level or nature of activity could impact the observed trends, potentially skewing toxicity levels and limiting the accuracy of conclusions drawn regarding the evolution of toxic content.

#### 2.3.3 Data quality and completeness
It is assumed that the Reddit dataset provided accurately represents the conversations within these communities without significant missing data or technical issues such as gaps in data collection. If there are gaps or data quality issues, it could lead to misrepresentation of toxicity trends, affecting both the insights and recommendations generated. 

## Section 3: Methodology
### 3.1 Technical Assumptions
In this subsection, we will outline key assumptions in our methodology. These assumptions ensure that our models and evaluation metrics are suitable and effective given our Reddit dataset’s characteristics and project goals.

#### 3.1.1 Transferability of model and relevance of jigsaw data for reddit toxicity classification
We assume that pretrained BERT models, fine tuned on the Jigsaw dataset, are suitable for toxicity classification in our Reddit dataset. This assumption is based on the idea that the language patterns and toxic discourse in the Jigsaw dataset share significant overlap with those in the Reddit dataset, allowing the model’s learned representations to generalise effectively. While Reddit’s tone and style may differ slightly, we expect the toxic language patterns to be sufficiently similar for the pretrained models to perform well in providing reliable classification on our Reddit data. Future validation steps such as testing a sample of labelled Reddit comments could confirm model accuracy on this specific domain, if necessary.

### 3.2 Data
All data preprocessing steps below were performed in the ```cleaning_and_sampling.ipynb``` notebook.

#### 3.2.1 Data Collection
The dataset that we will be analysing on consists of Reddit comments from 2020 to 2023 across three Singapore subreddits namely, r/Singapore, r/SingaporeRaw, and r/SingaporeHappenings. It includes nearly 5 million rows with 9 columns namely, ```text```, ```timestamp```, ```username```, ```link```, ```link_id```, ```parent_id```, ```id```, ```subreddit_id```, and ```moderation```. For our analysis, the primary columns used were ```text```, which contains the individual comments, ```timestamp```, which records the exact date and time each comment was sent, and ```subreddit_id```, indicating which subreddit each comment belongs to.

The Reddit dataset lacks toxicity labels which are essential for evaluation of classification models. Therefore, an external dataset was introduced, the Jigsaw Toxic Comment Classification Challenge dataset from Kaggle. This dataset provides Wikipedia comments labelled by human raters for toxicity, making it suitable for evaluating model performance on toxicity prediction. The dataset was labelled for various types of toxicity, such as: ```toxic```, ```severe_toxic```, ```obscene```, ```threat```, ```insult```, and ```identity_hate```. However in the context of our analysis, we would only be using the ```toxic``` column. Notably, the Jigsaw dataset includes comments made by users that contain both Singlish and curse words, aligning closely with the content in the Reddit dataset. This similarity supports the use of the Jigsaw dataset as a benchmark for selecting the most effective toxicity prediction model for our analysis.

#### 3.2.2 Data Cleaning
Firstly, rows containing removed or deleted comments were filtered out of the dataset, as the original text was not available for analysis. Next, we identified and removed the comments that were made by bots, as they are not considered as comments made by reddit users and often exhibit repetitive patterns. To identify these bot-generated comments, we extracted usernames and comments containing the term "bot", and ranked users by their total comment count. By reviewing comments associated with these usernames, several bot accounts surfaced, allowing us to replace their corresponding comments with ```NaN``` values. Rows with ```NaN``` values in either the ```text``` or ```timestamp``` columns were then removed. Lastly, duplicate rows, which we defined as rows that shared the same username, text, and timestamp, were deleted. This cleaning process resulted in a refined dataset containing approximately 4.5 million rows.

In the Jigsaw dataset, there were no ```NaN``` values, but the test labels included ```-1``` values, representing rows that were unlabelled. These rows were removed from the test set to ensure only labelled data was used for model evaluation.

#### 3.2.3 Feature Engineering
After data cleaning, data transformation was performed to structure the dataset for subsequent analysis. We first removed the ```link_id``` and ```moderation``` columns, as we will not be using them for our analysis. From the ```link``` column, the topic of the Reddit post to which each comment was replying to was extracted by extracting the text after the last forward slash in each link. Additionally, the timestamp column was split into separate ```date``` and ```time``` columns to facilitate temporal analysis. After these transformations, the original ```link``` and ```timestamp``` columns were removed from the dataset.

Next, basic text cleaning was applied to the text column to standardise the content. This included converting all characters to lowercase, limiting repeated letters to a maximum of two consecutive occurrences, removing HTML tags and URLs, and removing extra whitespaces. Given that the comments were from Singapore-focused subreddits, many contained Singlish terms. To address this, a custom list of Singlish stopwords was created to filter out singlish terms with minimal semantic value. Additionally, we noted that social media comments often feature abbreviations and slang, hence we created an Excel sheet to map and expand common abbreviations. Contractions were replaced with their full forms, while Chinese characters, emojis, and other symbols were removed from the text column to ensure consistency. Finally, rows with empty strings in the text column were removed from the dataset. The processed dataset was then saved as ```reddit_data_llm.csv``` for further analysis.

The same basic text cleaning steps were applied to the Jigsaw dataset to ensure consistency across both datasets. This included converting text to lowercase, limiting repeated letters, removing HTML tags, URLs, and extra whitespace, removing Singlish stopwords, processing contractions and abbreviations, as well as removing emojis and chinese characters. By applying identical preprocessing, the datasets were standardised, supporting consistent and reliable model evaluation results.

#### 3.2.4 Data Sampling
Given that the cleaned dataset contains approximately 4.5 million rows, running models on the entire dataset would be time-prohibitive. Therefore, we opted to work with a representative subset to achieve faster processing while maintaining accuracy.

To determine the optimal sample size, we applied Cochran’s sample size formula, selecting a 99% confidence level and a 1% margin of error. Using a population size of 4.5 million, the resulting ideal sample size was approximately 17,000 rows, which we rounded up to 20,000 for greater reliability.

To ensure representativeness, we used stratified sampling to select the 20,000 rows. We chose two characteristics for our stratification: quarter of the year and subreddit. Disproportionate sampling was done on the quarters of the year, which involved taking an equal number of rows from each quarter of the year, as the data will undergo quarterly temporal analysis. We also performed proportionate sampling for the subreddits, where the original subreddit proportions were preserved, ensuring that the subset closely reflects the characteristics of the full dataset.

### 3.3 Classification
The various classification models were explored and evaluated in the ```classification.ipynb``` notebook.

#### 3.3.1 Algorithms
The Jigsaw dataset contains comments from Wikipedia talk pages labelled for toxicity, with ```0``` denoting non-toxic comments, and ```1``` indicating toxic comments. To label our Reddit dataset, we used the Jigsaw dataset as a source of truth to compare the performances of classification models as the type of comments found were similar to the ones found in our Reddit dataset, containing singlish and slang. All models were evaluated on the Jigsaw dataset, which consists of approximately 220,000 rows. We selected a subset of 5% of the total dataset which is around 10,000 rows as testing data from the ```test_cleaned.csv``` dataset. A fixed random state was used during the selection to ensure reproducibility, such that the same rows can be selected each time the code is executed. All classification models were evaluated on this test set to ensure fair comparison.

##### 3.3.1.1 Pre-trained Models
We explored 3 BERT-based models that were pre-trained on the Jigsaw dataset, namely Toxic-BERT model, RoBERTa model and Dehate-BERT model. They are optimised for toxic language, general language understanding, and hate speech respectively.

##### 3.3.1.2 Custom Model
We also trained our own model on the Jigsaw dataset using the Logistic Regression Algorithm but with Naïve Bayes features.

This combines generative and discriminative classifiers where a Logistic Regression model is built over Naïve Bayes log-count ratios as feature values. It is similar to the Support Vector Machine with Naïve Bayes features, which has been shown to be a strong and robust performer over classification tasks. 

**Naïve Bayes Logistic Regression Model**
The Jigsaw dataset is originally split into training and testing datasets and they are first converted to a matrix of TF-IDF features, with the help of the TfidfVectorizer. Naïve Bayes features are then incorporated into these features before model training.

**Unbalanced vs balanced dataset**
As the Jigsaw training data is heavily imbalanced, with the ratio of toxic comments to non-toxic comments being 2:19, we tried to upsample the minority class in the training data to achieve class balance. Downsampling was not considered since that will result in a huge loss of information. 

|                          | Precision (for the toxic class) | Recall (for the toxic class) | AUC-ROC score |
|--------------------------|----------------------------------|------------------------------|---------------|
| Original unbalanced data | 0.60                            | 0.78                         | 0.9635        |
| Upsampled balanced data  | 0.50                            | 0.89                         | 0.9616        |

The table above summarises the results obtained when training was performed on both the original unbalanced dataset and an upsampled balanced version of the training data. We observed that upsampling decreases the AUC-ROC score slightly.

Therefore, we decided to not upsample the data, mainly because we felt that achieving a good balance between the recall and precision scores is important.

#### 3.3.2 Evaluation
After labelling the test set of the Jigsaw dataset using the different models, we checked against the truth label and calculated the AUC-ROC scores as shown in the table below.

**AUC-ROC Score (rounded to 3 s.f.)**
| Toxic-BERT | RoBERTa | Dehate-BERT | NB-log regression |
|------------|---------|-------------|-------------------|
| 0.974      | 0.976   | 0.843       | 0.962            |

As a higher AUC-ROC score indicates better performance of the classifier in distinguishing positive and negative classes, we concluded that the RoBERTa model, with the highest score, performed best in classifying comments.

The RoBERTa model outputs a probability score for each comment, indicating the likelihood of it being toxic. To effectively categorise comments as toxic or non-toxic, we needed to define an optimal threshold for these probability scores. Comments with a probability score above this threshold will be classified as toxic, while those below will be classified as non-toxic. 

For setting the threshold, we prioritised achieving a balanced precision and recall score. This approach minimises both false positives, non-toxic comments incorrectly labelled as toxic, and false negatives, toxic comments missed by the model, ensuring a more reliable dataset for trend analysis. By balancing precision and recall, we aimed to maintain a high recall rate, ensuring that a significant portion of toxic comments was captured, while reducing unnecessary false positives that could complicate the analysis.

We determined the optimal threshold score to be 0.936, which allowed us to classify comments effectively. This threshold selection process ensured the labelled Reddit dataset used was clean and well-suited for subsequent analysis.

### 3.4 Topic Modeling
For topic modelling, we focused on analysing comments which have been classified as toxic by our classification model. By performing topic modelling, we can uncover the underlying themes and patterns within the negative discourse on Reddit and subsequently better understand the reasons, triggers and contexts behind toxic behaviour on Reddit and the change in toxicity over the years. 

We chose 2 algorithms for Topic Modelling – Latent Dirichlet Allocation (LDA) and BERTopic. These models were chosen for their suitability to handle large and unstructured datasets like Reddit comments. We applied fine tuned versions of both models on the toxic comments and compared their coherence scores, diversity scores and topic labels to select the model that we would be using for further analysis of the dataset. 

The steps involving topic modelling can be found in the ```topic_modelling.ipynb``` notebook.

#### 3.4.1 Preprocessing 
We began by further preprocessing the toxic comments. Although our preprocessing steps are the same for both topic models (LDA and BERTopic), they are applied at different stages. In LDA, the steps are applied before fitting the toxic comments into the model, while in BERTopic, they are applied after clustering but before generating topic labels.

The preprocessing steps are as follows:
1. **Stopword Removal**: We manually identified filler words and curse words that do not contribute meaningful insights to the topics. Additionally, we incorporated common stopwords from sklearn’s ```ENGLISH_STOP_WORDS``` list. Removing these words allows the models to focus on more informative terms, enhancing their ability to detect patterns and relationships in the data.
2. **Removing Numbers and Punctuation**: We removed these elements as they typically do not add valuable information and may introduce noise.
3. **Lemmatization**: We applied lemmatization to reduce words to their root forms, which helps the models generalise better and ensures that the topic labels do not contain repetitive forms of the same word.

For BERTopic, these steps were applied after clustering to ensure that the model had full context for creating accurate embeddings and clusters. However, it was done before generating topic labels to prevent stopwords, punctuation, and numbers from appearing in the topic labels. This ensures cleaner and more informative labels.

#### 3.4.2 Latent Dirichlet Allocation (LDA)
LDA works by first randomly assigning each word in each comment to a random topic. Then, LDA iteratively refines the assignments by calculating the co-occurence of words within and across topics before reassigning the words to a topic. LDA iteratively refines topic assignments for each word in the dataset until it reaches a stable state where the topic distributions become more coherent and consistent.

#### 3.4.3 BERTopic
BERTopic leverages transformers and c-TF-IDF to create dense, meaningful clusters of topics. The algorithm converts documents into numerical embeddings optimised for semantic similarity, reduces dimensionality to handle high-dimensional data, and clusters the embeddings. It then combines the documents in each cluster, generates a bag-of-words representation, and uses c-TF-IDF to identify the most significant words for each cluster. The top words are selected to represent the core topics, making the results both interpretable and relevant.

#### 3.4.4 Evaluation
To evaluate the two algorithms, we looked at the diversity and coherence metrics as they are commonly used to evaluate topic models. Topic coherence evaluates how related the most representative words of a topic are. A high topic coherence score indicates that the words within a topic are meaningfully related, enhancing the interpretability of the topic and making it easier to assign a coherent label. Topic diversity score assesses how diverse or unique the topics generated by a topic model are. A high topic diversity score shows that the topic model covers a wide range of topics without over-emphasising on any particular topic. By optimising these two metrics, we can ensure that the topics generated are interpretable and insightful, while minimising duplicated topics. This can enable us to better understand the reasons and contexts behind these toxic comments. 

On top of quantitative metrics, we also manually analysed and compared the topic labels generated by the two topic models based on how interpretable and insightful they are.

##### 3.4.4.2 Choice of topic coherence metric
There are several common metrics for the calculation of topic coherence, such as _c_v_ , _u_mass_, _c_uci_ and _c_npmi_. The table below summarises their differences (Röder et al., 2015).

| Metric       | Segmentation    | Probability Calculation     | Confirmation Measure                                               |
|--------------|-----------------|-----------------------------|---------------------------------------------------------------------|
| c_v (between 0 and 1) | one-all       | Sliding window of size 110 | Indirect measures using Normalised Pointwise Mutual Information (NPMI) |
| c_umass (between -14 and 14) | one-preceding | documents                  | Logarithmic conditional probability                                 |
| c_uci        | one-one        | Sliding window of size 10   | Pointwise mutual information (PMI)                                  |
| c_npmi       | one-one        | Sliding window of size 10   | NPMI                                                                |

We chose _c_v_  over _c_uci_ and _c_npmi_ because indirect measures capture relations better than direct measures. We chose _c_v_ over _c_umass_ because _c_umass_ works based on document-based co-occurrence, which is more suitable for long and well-defined documents where topics are consistent throughout. On the other hand, _c_v_ uses sliding window-based co-occurrence, which focuses on local context and immediate word associations. Thus, it is more suitable for informal, short and multi-topic texts like our Reddit comments. Based on this, we think that _c_v_ is the best measure to use.

##### 3.4.4.4 Choice of topic diversity metric
Common topic diversity metrics include Pairwise Word Embedding Cosine Distance, Kullback-Leibler Divergence, and Topic Uniqueness Score (Silviatti, n.d.).

Pairwise Word Embedding Cosine Distance measures the similarity between words in different topics using cosine distance between their vector representations. Larger distances indicate more distinct topics. Kullback-Leibler Divergence measures differences between topic distributions, while the Topic Uniqueness Score evaluates how often a word appears in only one topic.

We chose Pairwise Word Embedding Cosine Distance for its suitability with embedding-based models like BERTopic, which capture semantic relationships. Kullback-Leibler Divergence is more suited for probabilistic models, and the Topic Uniqueness Score is better for specialised datasets with minimal topic overlap.

#### 3.4.5 Training
To fine-tune each topic model, we experimented with different sets of the model’s hyperparameters and selected the set that gives the most interpretable topic labels and the highest coherence and diversity scores.

##### 3.4.5.1 Training of the LDA model
For LDA, the most important hyperparameter to tune is the number of topics. A low number of topics may lead to broad, vague topics, while a high number may result in overly specific or redundant topics. Tuning this helps ensure that topics are interpretable and relevant to the dataset.

To do so, we varied the number of topics in each run, calculating the coherence score each time. By selecting the run with the highest coherence score (Figure 1), we can obtain the optimal number of topics to run for LDA, which is 10. 

![Figure 1](https://github.com/user-attachments/assets/fa93f760-4e35-4d32-8069-3a333df964ff)

**Figure 1:** Coherence Score Against Number of Topics for LDA

##### 3.4.5.2 Training of the BERTopic model
For BERTopic, we experimented with different embedding models and the most important hyperparameters which are the _n_neighbors_ and _n_components_ parameters in UMAP, the _min_cluster_size_ parameter in HDBSCAN, the _min_df_ parameter in the CountVectorizer as well as the _top_n_words_ parameter in the BERTopic.

The embedding models that we tried are the ```all-MiniLM-L6-v2``` model, which is the default embedding model, the ```paraphrase-MiniLM-L6-v2``` model and the ```all-mpnet-base-v2``` model. We ended up choosing the ```paraphrase-MiniLM-L6-v2``` model as it gives the most interpretable topic labels. 

The _n_neighbors_ parameter in UMAP is the number of neighbouring sample points used when making manifold approximation. Decreasing this value results in a more local structure, while increasing this value captures a more global structure, which often results in larger clusters being created. On the other hand, the _min_cluster_size_ parameter in HDBSCAN is the minimum size a final cluster can be which has a direct impact on the number of topics that will be generated. We tried various different sets of values for these two parameters and manually evaluated the topic labels based on interpretability.

The _n_components_ parameter in UMAP is the dimensionality of the embeddings after dimensionality reduction. This parameter is kept low to avoid the curse of dimensionality but we also experimented with slightly higher values from the range of 5 to 8 and observed whether the clustering improved by analysing the comments grouped under each topic. 

The _min_df_ parameter in the CountVectorizer, which represents how frequent a word must be before being added to our representation, is kept at 2. We chose to keep it low because our toxic comments dataset only consisted of around 8000 rows and furthermore, the stopwords are removed in this step. Therefore, the vocabulary size will not be too big, even if we keep the value at 2. 

The _top_n_words_ parameter in the BERTopic, which refers to the number of words per topic to be extracted. We kept it at 10 as the author recommended keeping this value between 10 and 20 to enhance the coherence of the most representative words of a topic. Moreover, the toxic comments are also quite short and are lemmatized so there may not be many distinct words that can represent a topic well. 

Some hyperparameters are kept as its default value. For instance, the metric parameter in UMAP, which refers to the method used to compute the distances in high dimensional space is kept as ‘cosine’ as our embeddings are high dimensional. 

#### 3.4.6 Model Selection
To select the better model between LDA and BERTopic, we compared the highest topic coherence scores and topic diversity scores obtained by the two models after hyperparameter tuning. We also manually compared the topic labels generated by them based on interpretability. 

| Model         | Coherence Score  | Diversity Score     |
|---------------|------------------|---------------------|
| BERTopic      | 0.5485           | 0.8675              |
| LDA           | 0.6614           | 0.7412              |

The table above summarises the scores obtained by the two topic models. Based on the table, we observe that LDA has a higher coherence score but a lower diversity score than BERTopic. 

![Figure 2](https://github.com/user-attachments/assets/30aa35a6-cbf4-4dc0-b368-bf1219b01fe5)

**Figure 2:** Topic labels generated by LDA after hyperparameter tuning

![Figure 3](https://github.com/user-attachments/assets/50015c7e-d26b-45fc-ad9a-583630b37adf)

**Figure 3:** Topic labels generated by BERTopic after hyperparameter tuning

From the above figures, we can see that the topic labels generated by BERTopic are much more interpretable than that of LDA, despite the fact that LDA has a higher coherence score than BERTopic. One reason for this could be that the words in the topic labels generated by LDA generally occur at a higher frequency so the co-occurrence of the words are naturally higher too. This also implies that LDA seems to only produce topics with words that co-occur but may not be meaningfully related, leading to topics that can be hard to interpret. This is unlike BERTopic, which uses transformer-based embeddings to capture deep semantic meaning within text. This allows it to capture relationships between similar words that may not co-occur frequently but are semantically related.

The non-interpretable topics generated by LDA could be also attributed to the length of the toxic comments. As the toxic comments are generally quite short, with the average length being 27 words, LDA which depends on word frequency patterns across documents has less context to work with, leading to less interpretable topics.

As our main aim of topic modelling is to gain insights from the toxic comments, we decided to select BERTopic as our model for further analysis, instead of LDA as it can generate better topics. Another reason for choosing BERTopic is the flexibility BERTopic offers since it allows users to customise each step of the algorithm.

## Section 4: Findings

### 4.1 Results
The goal of our project is to analyse toxic comments on Reddit to understand the increase in toxic content and identify actionable insights for mitigating toxicity. We first analysed the toxic comments in the dataset to examine trends in toxicity before applying topic modelling using BERTopic on the identified toxic comments to uncover the key drivers of toxicity on Reddit.

#### 4.1.1 Analysis on Toxicity Trends
Following the classification and labelling of toxic comments in the dataset, we began by conducting an analysis to examine overall trends in toxic behaviour across Singapore-focused subreddits over time. This step aims to highlight any general patterns in toxicity and provide context, before applying advanced techniques like topic modelling to uncover specific themes and drivers of toxic behaviour. 

The code for data exploration can be found in the ```analysis_of_toxicity_trends.ipynb``` notebook.

##### 4.1.1.1 Quarterly Trends in Percentage of Toxic Comments
Our first approach tracks the percentage of toxic comments per quarter to identify changes in toxic discourse within Singapore-focused subreddits. Given that our sample is approximately 5% of the total data, stratified with an equal number of comments each quarter, we focused on percentage-based trends to observe the relative increase or decrease in toxicity over time. This approach allowed us to gain insight into toxicity patterns without being influenced by variations in overall comment volume.

![Figure 4](https://github.com/user-attachments/assets/0fe8c622-c141-481a-ae57-49d0374c3991)

**Figure 4:** Percentage of Toxic Comments per Quarter




