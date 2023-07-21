# barotti-GeoLingIt2023

This repository contains the code for the paper "baρtti at GeoLingIt: Beyond Boundaries, Enhancing Geolocation Prediction and Dialect Classification on Social Media in Italy" accepted at the [GeoLingIt 2023 Shared Task](https://sites.google.com/view/geolingit).

## Table of Contents

- [Abstract](#abstract)
- [Data collection](#data-collection)
- [Task A](#task-a)
- [Task B](#task-b)
- [Experimental Setting](#experimental-setting)
  - [Harware Setting](#hardware-settings)
  - [Parameter Setting](#parameter-settings)
- [License](#license)
- [Contact Information](#contact-information)


## Abstract
The increasing usage of social media platforms has created opportunities for studying language use in various sociolinguistic dimensions. 
Italy, known for its linguistic diversity, offers a unique context for investigating diatopic variation, encompassing regional languages, dialects, and varieties of Standard Italian. 
This paper presents our contributions to the GeoLingIt shared task, focusing on predicting the locations of social media posts in Italy based on linguistic content. 
For Task A, we propose a novel approach, combining data augmentation and contrastive learning, that outperforms the baseline in region prediction.
For Task B, we introduce a joint multi-task learning approach leveraging the synergies with Task A and incorporate a post-processing rectification module for improved geolocation accuracy, surpassing the baseline and achieving first place in the competition.

## Data collection

Data augmentation plays a crucial role in our methodology, as it is implemented to address the class imbalance during the training process. 

In the initial data collection phase, we obtain a substantial amount of regional dialect data from various online sources, including [Italian Wikipedia](https://it.wikipedia.org) ([dump](https://dumps.wikimedia.org/itwiki/20230501/)) and [Dialettando](https://dialettando.com) (accessed on April 2023).

For Wikipedia, we leverage some specific language editions, e.g., [Neapolitan](https://nap.wikipedia.org/), [Piedmontese](https://pms.wikipedia.org/), etc.

## Task A
The goal of task A is to identify the origin region of a given tweet.

- **Pre-training with contrastive learning and data augmentation.** In the initial data collection phase, we obtain a substantial amount of regional dialect data from [Italian Wikipedia](https://it.wikipedia.org) and [Dialettando](https://dialettando.com). We then pre-process them, obtaining an expanded dialect vocabulary that is utilized for the purpose of data augmentation. We adopt a substitution approach to words in tweets representing language variations of Italy to build an augmented version of the original dataset. Each tweet belonging to a region is augmented by randomly replacing words that are contained in the dialect vocabulary with other words from the same region, with a random probability $p$. Regarding the contrastive learning strategy, we pre-train the model to enhance its ability to discern whether two tweets belong to the same region. During this preliminary training phase, the model learns to differentiate between tweet pairs and their corresponding regional affiliations. In this approach, we randomly select a sample from the dataset to serve as an anchor. We then create a positive data point by augmenting the anchor with words from the same region, and a negative sample by augmenting the anchor with words from a different region. By incorporating these positive and negative samples, the model is trained to distinguish between the regional affiliations of different tweets.
  
- **Entropy-based Ensemble.** We empirically observed that a simple logistic regression model achieves good performance in words of F1 score on various minority classes. We attribute this to a lower model capacity, reducing the amount of overfitting that may occur in minority classes. To leverage this insight, we propose using an exclusive class assignment mechanism that uses the confidence of the BERT-based model. In other words, when the BERT-based prediction is made with low confidence, we replace the overall prediction with the one made by the logistic regression, if the latter's confidence is higher. We estimate the confidence of the BERT-based model using the entropy of its predicted probabilities. Lower entropy is associated with high certainty (i.e., the model predicts one class with high probability and all others with a low one), and vice-versa.


## Task B
The objective of this task is to automatically determine the geographical coordinates (latitude and longitude) of the origin of a tweet.

- **Multi-Task fine-tuning.** Recognizing the strong correlation between Task A and Task B, we adopt a multi-task learning approach to tackle them jointly. In the multi-task learning setup, we build a two-head model by adding to the classification a regression layer, enabling the estimation of coordinates together with the regional classes. The model is thus trained to simultaneously learn both the geographical location and the class of the tweet. To achieve this, we optimize the model using a weighted combination of loss functions. For Task A, we aim to maximize the F1 score by minimizing the corresponding cross-entropy loss. For Task B, we minimize the Haversine distance by minimizing the mean squared error (MSE) loss, which helps to reduce the difference between the predicted and target coordinates.

- **"Beyond-Boundaries" Multi-Task fine-tuning.** In addition to the joint task learning, we introduce a rectification module to refine the model's predictions. This module leverages the geographical domain knowledge that dictates that tweets are expected to be found within the territorial confines of the country. In other words, it ensures that tweets are geographically located within the national borders, specifically on land rather than in the sea. Coordinates that fall outside of these constraints are adjusted so as to be set to the closest point within the boundaries. By incorporating these techniques, we aim to enhance the model's performance in both identifying the geographical origin of a tweet and classifying its region. We enforce this constraint as a post-processing step, where coordinates are projected onto a high-resolution map of Italy (with a granularity of 1.5 km). Points that fall outside of the boundaries of Italy are moved to the closest point within the country.

- **Continuous Learning.** For the sake of completeness, we conclude the analysis with an approach that merges the pre-training with contrastive learning and data augmentation strategy with the multi-task fine-tuning scheme. We specifically train the model in a multi-task manner, starting from the pre-trained model that underwent contrastive learning and data augmentation and was fine-tuned for task A.
  
## Experimental setting

### Hardware settings
All experiments were conducted on a private workstation with Intel Core i9-10980XE CPU, 1 $\times$ NVIDIA RTX A6000 GPU, 64 GB of RAM running Ubuntu 22.04 LTS.

### Parameter settings

- Batch Size: 8
- Max Number of Epochs: 10
- Learning Rate: 5e-5
- Optimizer: AdamW
- Weight Decay: 0.01
- Warmup Ratio: 0.06
- Gradient Accumulation Steps: 1

<!-- 
## Citation

If you use this code, please cite the following paper:

```
``` -->

## License
This code is released under the Apache 2.0 license. See the [LICENSE](LICENSE) file for more details.

## Contact Information
For help or issues using the code, please submit a GitHub issue.

For other communications related to this repository, please contact [Alkis Koudounas](mailto:alkis.koudounas@polito.it).
