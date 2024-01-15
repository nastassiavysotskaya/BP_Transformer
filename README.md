# BP_Transformer
Welcome to the code repository of the paper

> TRANSFORMING CARDIOVASCULAR HEALTH: A TRANSFORMER-BASED APPROACH TO CONTINUOUS, NON-INVASIVE BLOOD PRESSURE ESTIMATION VIA RADAR SENSING

accepted at ICASSP 2024 [1].

As detailed in the paper, the model is pretrained on arterial blood pressure data first to get a robust initial latent space.
Only then is it finetuned on radar obtained skin displacement for predicting continuous beat-to-beat blood pressure value pairs.

# Data
### Prerequisite: Download PulseDB dataset
To pretrain the model on the arterial blood pressure (ABP) data on the carefully curated [PulseDB](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2022.1090854/full) dataset,
please follow the detailed and easy-to-follow instructions on their [github repository](https://github.com/pulselabteam/PulseDB/) to obtain the data. <br>
When you do use their data, please make sure to cite them accordingly [[3]](https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2022.1090854/full).

### Radar Data will not be made publicly available!


### Building feature database
As detailed in our work [[2]](https://www.mdpi.com/1424-8220/23/8/4111), we extract features per pulse wave for a pulse wave analysis-based blood pressure predicting approach.<br>
As such, 21 carefully curated amplitude- and timing-based features are extracted per wave. <br>
The physiological connection between these features and blood pressure is detailed in [[2]](https://www.mdpi.com/1424-8220/23/8/4111).
If you decide to use the feature extraction for your own work, please cite our previous work [[2]](https://www.mdpi.com/1424-8220/23/8/4111) where all of this is detailed.


To extract the ABP features, please run *create_dataset/build_feature_database_ABP.py*.
It will extract the features and save them to a .csv file. 
This will make training faster compared to online feature extraction.

# Training the model
To pretrain the model on ABP data, run *train_BP_prediction_models/pretrain_model_on_ABP_data.py*. 

Radar data will not be made available. If you decide to apply this code to your own radar data, please make sure that you extract the skin
displacement data according to the description in our work. If you use other sensing modalities, such as PPG, the same algorithm can be applied.
If you publish something using our proposed feature extracting pipeline, please cite our work[[2]](https://www.mdpi.com/1424-8220/23/8/4111).
If you decide to use our network architecture, please cite our work [1].

When you extract the time-series representation of your own radar data, make sure to extract features using our feature_utils, the same way you are extracting ABP features.
Once you have extracted the features, you can finetune the model on radar data using *train_BP_prediction_models/train_model_on_radar_data.py*.

To test the model, you first execute *train_BP_prediction_models/test_radar_model.py* and then analyse the resulting metrics using *train_BP_prediction_models/analyse_results.py*.

# References

> [1] Vysotskaya N, Maul N, Fusco A, Hazra S, Harnisch J, Arias-Vergara T, Maier A. TRANSFORMING CARDIOVASCULAR HEALTH: A TRANSFORMER-BASED APPROACH TO CONTINUOUS, NON-INVASIVE BLOOD PRESSURE ESTIMATION VIA RADAR SENSING. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE.
> 
> @Article{vysotskaya24icassp,<br>
    AUTHOR = {Vysotskaya, Nastassia and Maul, Noah and Fusco, Alessandra and Hazra, Souvik and Harnisch, Jens and Arias-Vergara, Tomás and Maier, Andreas},<br>
    TITLE = {TRANSFORMING CARDIOVASCULAR HEALTH: A TRANSFORMER-BASED APPROACH TO CONTINUOUS, NON-INVASIVE BLOOD PRESSURE ESTIMATION VIA RADAR SENSING},<br>
    booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},<br>
    year={2024},<br>
    organization={IEEE}<br>
}

> [2] Vysotskaya N, Will C, Servadei L, Maul N, Mandl C, Nau M, Harnisch J, Maier A. Continuous Non-Invasive Blood Pressure Measurement Using 60 GHz-Radar—A Feasibility Study. Sensors. 2023; 23(8):4111. https://doi.org/10.3390/s23084111
> 
> @Article{s23084111,<br>
    AUTHOR = {Vysotskaya, Nastassia and Will, Christoph and Servadei, Lorenzo and Maul, Noah and Mandl, Christian and Nau, Merlin and Harnisch, Jens and Maier, Andreas},<br>
    TITLE = {Continuous Non-Invasive Blood Pressure Measurement Using 60 GHz-Radar&mdash;A Feasibility Study},<br>
    JOURNAL = {Sensors},<br>
    VOLUME = {23},<br>
    YEAR = {2023},<br>
    NUMBER = {8},<br>
    ARTICLE-NUMBER = {4111},<br>
    URL = {https://www.mdpi.com/1424-8220/23/8/4111},<br>
    PubMedID = {37112454},<br>
    ISSN = {1424-8220},<br>
    DOI = {10.3390/s23084111}<br>
}


> [3] Wang, W., Mohseni, P., Kilgore, K. L., & Najafizadeh, L. (2023). PulseDB: A large, cleaned dataset based on MIMIC-III and VitalDB for benchmarking cuff-less blood pressure estimation methods. Frontiers in Digital Health, 4, 1090854.
> 
> @ARTICLE{10.3389/fdgth.2022.1090854,<br>
>   AUTHOR={Wang, Weinan and Mohseni, Pedram and Kilgore, Kevin L. and Najafizadeh, Laleh},<br>
>   TITLE={PulseDB: A large, cleaned dataset based on MIMIC-III and VitalDB for benchmarking cuff-less blood pressure estimation methods}, <br>
>   JOURNAL={Frontiers in Digital Health},<br>
>   VOLUME={4},<br>
>   YEAR={2023},<br>
>   URL={https://www.frontiersin.org/articles/10.3389/fdgth.2022.1090854}, <br>
>   DOI={10.3389/fdgth.2022.1090854}, <br>
>   ISSN={2673-253X},   
}