#  FlowTransformer 2023 by liamdm / liam@riftcs.com

import os

import pandas as pd

from framework.dataset_specification import NamedDatasetSpecifications, DatasetSpecification
from framework.enumerations import EvaluationDatasetSampling
from framework.flow_transformer import FlowTransformer
from framework.flow_transformer_parameters import FlowTransformerParameters
from framework.framework_component import FunctionalComponent
from implementations.classification_heads import *
from implementations.input_encodings import *
from implementations.pre_processings import StandardPreProcessing
from implementations.transformers.basic_transformers import BasicTransformer
from implementations.transformers.named_transformers import *

encodings = [
    NoInputEncoder(),
    RecordLevelEmbed(64),
    CategoricalFeatureEmbed(EmbedLayerType.Dense, 16),
    CategoricalFeatureEmbed(EmbedLayerType.Lookup, 16),
    CategoricalFeatureEmbed(EmbedLayerType.Projection, 16),
    RecordLevelEmbed(64, project=True)
]

classification_heads = [
    LastTokenClassificationHead(),
    FlattenClassificationHead(),
    GlobalAveragePoolingClassificationHead(),
    CLSTokenClassificationHead(),
    FeaturewiseEmbedding(project=False),
    FeaturewiseEmbedding(project=True),
]

transformers: List[FunctionalComponent] = [
    BasicTransformer(2, 128, n_heads=2),
    BasicTransformer(2, 128, n_heads=2, is_decoder=True),
    GPTSmallTransformer(),
    BERTSmallTransformer()
]



pre_processing = StandardPreProcessing(n_categorical_levels=32)

# Define the transformer
ft = FlowTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[0],
                     sequential_model=transformers[0],
                     classification_head=classification_heads[0],
                     params=FlowTransformerParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))





# Load the specific dataset
dataset_name = "DIAD"
dataset_path = '\\\\wsl.localhost\\Ubuntu\\home\\ubuntu\\DatasetFlow\\merged_binary_dataset.csv'
eval_percent = 0.01
eval_method = EvaluationDatasetSampling.LastRows

dataset_specification = DatasetSpecification(
    include_fields=[
        'Flow duration', 'total Fwd Packet', 'total Bwd packets', 'total Length of Fwd Packet',
        'total Length of Bwd Packet', 'Fwd Packet Length Min', 'Fwd Packet Length Max',
        'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Min',
        'Bwd Packet Length Max', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
        'Flow IAT Min', 'Fwd IAT Min', 'Fwd IAT Max', 'Fwd IAT Mean', 'Fwd IAT Std',
        'Fwd IAT Total', 'Bwd IAT Min', 'Bwd IAT Max', 'Bwd IAT Mean', 'Bwd IAT Std',
        'Bwd IAT Total', 'Fwd PSH flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
        'Fwd Header Length', 'Bwd Header Length', 'FWD Packets/s', 'Bwd Packets/s',
        'Packet Length Min', 'Packet Length Max', 'Packet Length Mean', 'Packet Length Std',
        'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
        'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWR Flag Count', 'ECE Flag Count',
        'down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
        'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg', 'Bwd Bytes/Bulk Avg',
        'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
        'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Fwd Init Win bytes', 'Bwd Init Win bytes',
        'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Min', 'Active Mean', 'Active Max',
        'Active Std', 'Idle Min', 'Idle Mean', 'Idle Max', 'Idle Std'
    ],
    categorical_fields=['Src Port', 'Dst Port', 'Protocol'],
    class_column='Label',
    benign_label='Benign'
)



ft.load_dataset(dataset_name,
                dataset_path,
                dataset_specification,
                evaluation_dataset_sampling=eval_method,
                evaluation_percent=eval_percent)

# Build the transformer model
m = ft.build_model()
m.summary()

# Compile the model
m.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'], jit_compile=True)

# Get the evaluation results
eval_results: pd.DataFrame
(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=5, steps_per_epoch=64, early_stopping_patience=5)


print(eval_results)