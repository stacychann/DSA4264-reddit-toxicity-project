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

#### 3.1.2 Comparability of models using AUC-ROC scores
It is assumed that the AUC-ROC scores are reliable performance indicators across the various classification models explored, BERT-based models and Naive Bayes logistic regression model, for toxicity classification. We believe that the AUC-ROC metric effectively captures the models’ discriminatory power in distinguishing between toxic and non-toxic comments, making it appropriate for comparative evaluation. The model with the highest AUC-ROC score will be selected for labelling our Reddit dataset.

#### 3.1.3 Topic coherence and interpretability
We assume that the topics discovered through the topic modelling process will be coherent and interpretable. Specifically, we expect that the most significant topics will correspond to clear themes related to toxic discourse, such as use of aggressive language, insults or polarised opinions. If the generated topics are too abstract or difficult to interpret, we may need to refine preprocessing steps or adjust model parameters to enhance topic clarity.

#### 3.1.4 Effectiveness of preprocessing for topic clarity
We assume that the text preprocessing steps such as stopword removal and tokenisation, are sufficient for enhancing the clarity of topics discovered. If preprocessing is inadequate, for instance omitting domain-specific stopwords or failing to handle noisy data, the resulting topics may be less meaningful. This could impact the quality of insights derived from topic modelling, and further adjustments may be necessary.

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




