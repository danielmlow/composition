# composition
Study of semantic compositionality by comparing text categorization models 

contact: dlow@g.harvard.edu

## Ongoing manuscript on OverLeaf (already sent you a link)


**Clone repository:**
``` 
git clone https://github.com/danielmlow/composition.git
```

**Run this in terminal so the models in ./models/ with import modules like config.py**
```
export PYTHONPATH="${PYTHONPATH}:/path/composition"
```

**Dependencies**
* python3
* pandas
* numpy
* keras
* sklearn
* tensorflow

# To do
- I put all models within the dir /models/. So fix paths from being in the main dir to that one. 
- Edit all files, add more comments.
- Add dependencies

# Files
- MA_3120155_DM_Low

        Thesis


- 6_38_full_article_6000

        dataset. see thesis.

- gensim_it_w2v.pkl

        Needed to run cnn or lstm. Ask me for it.


- Xtrain_w2v_mean.npy (832MB) Xtest_w2v_mean.npy (184MB) 

        Request if you want to run any of the logreg_w2v models.

- stopwords-it2.txt

        These are stopwords which can be removed from avg. word embeddings. 


### Plotting & Analysis

- config.py

        Here you set paths for local and cluster, and other configuration options for your model.

- data_helpers.py

        Preprocessing and other useful functions.


- evaluate_sentences.py

        Include sentences in txt, csv or manually in ./evaluated_sentence/ directory and and it will output vectors from the cnn model to the ./evaluated_sentences/model_name_date_time/ directory (need to adjust code for other types of models).


- plot_outputs.py
    
        Plotting classification report, confusion matrix, learning curve 


- rsa.py

        After training a model, I plotted RSMs/RDMs with dendrograms between the sentences. 

- rsm.py

        Used more in thesis. More refined.

- rsa_similarity_experiment.py

        Functions to do RSA between feature vectors of different models and human responses

- corr_between_layers.py

        To build RSMs at each layer and then correlate them, to see how similar/redundant layers are. 

- probing.py

        Evaluate what's in the vectors by probing (see Sent Eval and thesis).

- rouge.py

        Rouge is a nice metric to see whether two sentences are superficially similar. We used it to show the layer1 in the CNN has higher rouge than layer 3 (where sentences are similar because they belong to the same category), proving the CNN is doing abstraction. See thesis and 

https://github.com/pltrdy/rouge

- text_categorization_experiment.py

        Some code for Experiment 3: Human Text Categorization (see thesis)


### /Models

##### Main models:
- all "*.sh" files are files to run .py files on the cluster. Each .py has one.

- cnn41_final.py
    
        Final CNN model in thesis. 

- lstm18.py

        LSTM final


- logreg_w2v.py
        
        Logistic regression using avg. word embeddings for each sentence. 


- logreg_tfidf.py

        Log regression using tfidf.

##### Secondary models:

- *_6.py

        Each model was trained on 6 categories to compare to Experiment 2: Human Similarity Judgments. For logreg model, this could be tf-idf or avg. word embedding model.

- *_4.py

        These were also run on 4 categories for Experiment 3: Human Text Categorization (see thesis).

- *_crossvalidation.py

        These files are cross validation on the train+validation sets to see how much variance existed between the folds. Then the above models were ran training on the train+validation sets and testing on the test set. 

##### Evaluating models:

- *_wo_w2v.py

        without word2vec to see effect of using pre-trained embeddings. 
    
- scrambled_both.py

        You randomize the word embeddings to see if what it's learning and if there is a zero effect of having sentence of different lengths + zero-padding.


### /outputs/

Example of the log files these models create. See thesis for results. The directory /final_model/ that I have locally has many of the weights (heavy files) and results. 


### final_experiment_daniel

Here you have the stimuli, psychtoolbox code, and more regarding the fMRI experiment. 

### Stimuli Design

- cluster_layers.py

        See Material section in thesis or manuscript. We clustered sentence-feature vectors from layer 1 and layer 3. Within each cluster,we chose the top 20 silhouette scores to obtain the mostrepresentative sentences from each cluster.       

- stimuli_final.py

        Here I create stimuli from all sentences in the dataset. 14 and 15 words that follow a certain similarity and dissimilarity pattern from layer 1 to layer 3 in the CNN. 


### survey

        Survey we did to see which sentence were most understandable. 

### /other_analyses

- architecture.xls

        Explains why we chose 14 and 15 words graphically.

- other directories are things we tried and didn't work  


