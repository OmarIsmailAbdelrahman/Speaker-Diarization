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
10. we integrated the ASR model which we trained to have 8.9 WER with nemo offline diarization
11. we tried to fix diart online problem by changing the embedding and clustring algorithm *Faild*
