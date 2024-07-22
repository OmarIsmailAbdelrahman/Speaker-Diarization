# Speaker-Diarization

1. we tried nemo offline diarization model
2. we tried to train nemo offline diarization it but *failed no change*
3. we created a custom dataset using SADA arabic dataset and created different configuration of this dataset from changing the overlapping duration and probability occurance and silance between audio files
4. we tried pyaanote offline diarization model and compared between it and nemo, and the results show that nemo is better "in paper nemo is better in clustrering and embedding model"
5. we found a problem in nemo clustring algorithm, which returns a single cluster and tried to fix it *give the code and example and test the difference between the original and modified code* ???
6. we tried different configuration on nemo offline diarization model, configuration 1.rp_threshold 2.sigmoid_threshold 3.different scales and weight *show results*
7. Nemo online *Faild*
8. Diarat Online based on pyaanote diarization and it worked but have problem with determining speakers
9. we tried to modifiy offline nemo to have a speaker linking *identify same speaker in different files*
10. we trained ASR model and achiverd WER 8.9 and tried different tokenize 
11. we integrated the ASR model which we trained to have 8.9 WER with nemo offline diarization
12. we tried to fix diart online problem by changing the embedding and clustring algorithm *Faild*
13. converted online diart from reading microphone to steaming files "add if possible that this allowed us to check for the DER and measure perofrmance"

## ASR Model Improvement
The unigram tokenizer based conformer which achieved  `11.9 MLD` used in phase 1 was using 32 precision which took double the training time of 16 precision model with no significant effect on the results. Therefore, we decided to use `16 precision` models in this phase.

### Phase 1 Unigram tokens
These was the tokens in phase 1 model which did not include `<fill>` `<overlap>` `<laugh>` tokens but instead used their characters as tokens which was incorrect. Also, the tokens included whole words which had more than 3 characters.
<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/81030289/f9a36541-1f66-4047-9d98-83fd6380af52" alt="Conformer CTC on NeMo"/>
</p>

### Improvements on Unigram Tokenizer
- Add the missing tokens and thus removing the english characters from the model tokens.
- Limit `max token length` to 2 characters to prevent whole words from being added to the model vocabulary
<p align="center">
  <img src="https://github.com/user-attachments/assets/4ad45c4a-a5ba-438f-ae58-ca5192caf08c" alt="Improved unigram tokenizer"/>
</p>

#### challenges
- Reducing precision to 16 made the model unstable when using the same `learning rate` as 32 version. To make the training stable the learning rate had to be reduced to 0.8. However, lowering it caused the loss and `WER` to plateau in an early stage of training compared to the char based moel.
<p align="center">
  <img src="https://github.com/user-attachments/assets/3ca85c19-9390-450b-80f0-9dc22c460d5a" alt="16 unigram loss and wer"/>
</p>

### Char Based Tokenizer
- We continued training of the char based model with the following tokens
<p align="center">
  <img src="https://github.com/OmarIsmailAbdelrahman/MTC-Competiton/assets/81030289/7efadcd7-5f05-4202-b299-31b357309eca" alt="Conformer CTC on NeMo"/>
</p>

- Achieved `8.9 MLD` which is lower than what was achieved in phase 1 by 2%

<p align="center">
  <img src="https://github.com/user-attachments/assets/33697d5b-6381-478a-87e9-ccadfe05563e" alt="New wer"/>
</p>

## Offline Speaker Diarization Experiments
A typical speaker diarization pipeline consists of the following:
- Voice Activity Detector (VAD): detects the presence or absence of speech to generate segments for speech activity from the given audio recording.
- Segmentation: Further breaks down the detected speech segments into smaller segments. This step ensures that each segment is small enough for accurate speaker embedding extraction.
- Speaker Embedding Extractor: extracts speaker embedding vectors containing voice characteristics from raw audio signal.
- Clustering Module: Groups the extracted speaker embeddings into clusters, where each cluster represents a unique speaker. This step helps to identify and differentiate between multiple speakers in the audio.

### We attempted to use the following pipelines:
#### pyannote
 - Vad model: `pynet`
 - Embedding extractor model: `wespeaker-voxceleb-resnet34-LM`
 
Speaker recognition model based on the ResNet34 architecture, trained on the VoxCeleb dataset, and utilizing a linear margin loss function to enhance its performance in distinguishing between different speakers. Margin-based loss functions are designed to improve the discriminative power of the model by increasing the margin between different classes in the feature space.
 - Agglomerative clustering: Hierarchical clustering method used to group objects based on their similarities. It is a bottom-up approach where each object starts as its own cluster, and pairs of clusters are merged as one moves up the hierarchy

#### Nemo
- Vad model: `MarbelNet`
- Embedding Extractor: `Titanet-Large`
- Clustering Model: `NME-SC`
- Neural Diarizar: `MSDD`


We decided to use nemo for the following reasons: 
- Its embedding shows better `EER` results compared to `wespeaker-voxceleb-resnet34-LM` [[1]](#references). `Wespeaker-voxceleb-resnet293-LM` achieved better results than `Titanet-large`, however this is due to it having more parameters which would increase inference time [[1]](#references).
- Better clustering model, which is explained bellow.

##### NME-SC [[2]](#references)
- The new framework estimates the row-wise binarization threshold pp and the number of clusters kk using the NME value derived from the eigengap heuristic.
- The process involves creating an affinity matrix with raw cosine similarity values, binarizing it, symmetrizing it, computing the Laplacian, performing SVD, and calculating the eigengap vector.
- The NME value `gp` is used to find the optimal `p` and `k` number of clusters, with the ratio `r(p)` = `r / gp` serving as a proxy for the diarization error rate (DER).
- `p` value should be minimized to get an
  accurate number of clusters, while the `gp` value should be
  maximized to get the higher purity of clusters. Thus,
  the ratio `r(p)` = `p / gp` is calculated to find the best `p` value by getting a size of the `p` value in proportion to `gp`.
##### MSDD [[3]](#references)
<p align="center">
  <img src="https://github.com/user-attachments/assets/e130e28a-1858-483d-9593-6d52485cf742" alt="MSDD"/>
</p>

- The MSDD approach offers overlap-aware diarization, flexibility in speaker numbers, and improved performance without extensive parameter tuning. It can handle a variable number of speakers without being constrained by a fixed number during training. 
- The multi-scale approach is fulfilled by employing multi-scale segmentation and extracting speaker embeddings from each scale.
- When combining the features from each scale, the weight of each scale largely affects the speaker diarization performance.
- The final speaker labels are estimated using the context vector
<p align="center">
  <img src="https://github.com/user-attachments/assets/deaf8724-6b84-460a-8133-de1c03a2ae14" alt="context vector"/>
</p>

##### Challenges
- sometimes the configuration require to be set hard-coded because the configuration is not always set the same across the system, example "enhanced_count_thres" is always set to 80
- Nemo NME-SC implementation sometimes returned wrong number of clusters: tested enhanced_count_thres raneg from 0 to 100, this variable and min_samples_for_nmesc with default = 6 "minimum number of samples required for NME clustering" if either of them are higher than embedding it calls getEnhancedSpeakerCount, which adds dummy to add noise to the clustering algorithm, the mean of the number of embedding is above 40 thus most of the time the function is called because of the default value of 80, and we checked error between the correct number of cluster and the predicted number of clusters and the average predicted number of clusters which resulted into error of 1.535, and average of prediction 1.406 on dataset of average X clusters, which means that the cluster initialization fails to predict the correct number most of the time and predict a single cluster, this have a better DER result than setting it to 40, which have error of 1.605, and 0 with error of 2.284, but having a lower DER rate doesn't mean that it is a good thing because on dataset that always have more than 1 speaker means that the initalization can't cluster correctly and just consider every audio as a single cluster, thus problem is probabily caused by using the dummy clusters, the default value is 3 dummy clusters and this results to have a 4 clusters in total most of the time.
- and removing the number of dummy clusters makes the model highly unstable? 

##### Experiments
- Tuned the `rp` and `sigmoid` thresholds to arabic sadadest to optimizer the model performance on arabic speech.
- After testing values for `rp` from 0.03 to 0.5, we found `rp` = 0.25 gave the best `DER` results.
- Tested  the following range of values 0.5 <= `sigmoid threshold` <= 0.9 showed no significant improvement compared to default value of 0.75.
- fine-tuned the MSDD module but it showed no improvement.

##### Results
<p align="center">
  <img src="https://github.com/user-attachments/assets/b76d786c-8569-4a29-a884-30a7b852c124" alt="context vector"/>
</p>


## Online Speaker Diarization Attempt
### diart
Diart by default uses pyannote pipeline but adds online streaming and buffering functionality.
- Integrate Nemo ASR model with diart 
- Send audio as 2 seconds chunks, transcribe and dairize it then send next chunk.

| Enhanced Threshhold | Dummy Clusters | Mean Error of Number of Clusters | Diarization ER |
|---------------------|----------------|----------------------------------|----------------|
| 40                  | 1              | 8.655333333333333                | 0.4533         |
| 80                  | 1              | 19.493333333333332               | 0.6156         |
| 40                  | 2              | 8.084                            | 0.4489         |
| 80                  | 2              | 18.694666666666667               | 0.6085         |
| 40                  | 3              | 1.605                            | 0.3486         |
| 80                  | 3              | **1.5353333333333334**           | **0.3222**     |
| 40                  | 4              | 7.044666666666667                | 0.4388         |
| 80                  | 4              | 1.5746666666666667               | 0.3260         |
| 40                  | 5              | 1.6606666666666667               | 0.3532         |
| 80                  | 5              | 1.5886666666666667               | 0.3275         |


### Challenges
the model couldn't keep the profile of the speakers, and in long run the model starts to hallucination and keep predicting the same words
### Custom Diart Pipeline
  To solve this problem we build a custom pipeline that uses pyaanote VAD model then fed the output to TitaNet from Nemo for better embeddings, then we tried two different methods for Clustring, the first is using the online clustring class of nemo, and building our clustring model, "continue after building experements", and for easier debug we changed the pipeline to stream files instead from taking audio from microphone, this alows for measureing performance and consistance input.
## References
[1] S. Wang, Z. Chen, B. Han, H. Wang, C. Liang, B. Zhang, X. Xiang, W. Ding, J. Rohdin, A. Silnova, Y. Qian, and H. Li, "Advancing speaker embedding learning: Wespeaker toolkit for production first-line systems," *Neurocomputing*, vol. 559, pp. 125892, 2023. Available: [https://doi.org/10.1016/j.specom.2024.103104](https://doi.org/10.1016/j.specom.2024.103104).

[2] T. J. Park, K. J. Han, M. Kumar, and S. Narayanan, "Auto-Tuning Spectral Clustering for Speaker Diarization Using Normalized Maximum Eigengap," arXiv preprint arXiv:2003.02405, 2020. Available: https://arxiv.org/abs/2003.02405.

[3] T. J. Park, N. R. Koluguri, J. Balam, and B. Ginsburg, "Multi-scale Speaker Diarization with Dynamic Scale Weighting," arXiv preprint arXiv:2203.15974, 2022. Available: https://arxiv.org/abs/2203.15974
