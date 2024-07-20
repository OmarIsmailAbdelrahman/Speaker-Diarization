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
- Limit `max token length` to 2 tokens to prevent whole words being added to the model vocabulary
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

