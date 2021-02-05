# PHASE
Repository for the PHASE (PHysiologicAl Signal Embeddings) project.

The data used for the majority of experiments is not publicly available due to patient privacy concerns, with the exception of [https://mimic.physionet.org/](MIMIC-III) which is available under appropriate data use agreement.  As such we have omitted most of the processing code and organize the repositories for data and experimental results as follows:

    ./upstream_embedding/
        models/  (where upstream embedding models are saved)
            200epochs/
                autoencoder/
                hypoc/
                hypot/
                hypox/
                minimum5/
                nextfive/
                
    ./downstream_prediction/
        results/ (where evaluations of downstream prediction models are saved)
        data/    (where the data is saved)
            desat_bool92_5_nodesat/ (hypoxemia)
            etco235/                (hypocapnia)
            med_phenyl/             (phenylephrine)
            nibpm110/               (hypertension)
            nibpm60/                (hypotension)
            

## Training upstream embedding models

We train upstream LSTM embedding models in `experiments/upstream` using Keras with a Tensorflow backend.  In particular we include examples of several versions of embedding models including per-signal LSTM embedding models (with a variety of self-supervised tasks), a joint LSTM embedding model, and fine-tuned per-signal LSTM embedding models.

## Training downstream prediction models

For downstream prediction, we evaluate a number of embedding types utilizing gradient boosted trees (XGBoost), and multi-layer perceptrons (Keras with Tensorflow backend).  In addition, we train an LSTM on raw signal data (only processed to the extent of normalization).  Finally, we include an experiment with heterogenous sets of features, where we assume the downstream target hospital only has a subset of the features available in the upstream source hospital.

