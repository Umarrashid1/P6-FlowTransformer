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
dataset_path = 'diad_balanced.csv'
eval_percent = 1.0
eval_method = EvaluationDatasetSampling.RandomRows

dataset_specification = DatasetSpecification(
    include_fields=[
        'Flow_Duration', 'Total_Fwd_Packet', 'Total_Bwd_packets', 'Total_Length_of_Fwd_Packet',
        'Total_Length_of_Bwd_Packet', 'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min',
        'Fwd_Packet_Length_Mean', 'Fwd_Packet_Length_Std', 'Bwd_Packet_Length_Max',
        'Bwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean', 'Bwd_Packet_Length_Std',
        'Flow_Bytes_per_s', 'Flow_Packets_per_s', 'Flow_IAT_Mean', 'Flow_IAT_Std', 'Flow_IAT_Max',
        'Flow_IAT_Min', 'Fwd_IAT_Total', 'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max',
        'Fwd_IAT_Min', 'Bwd_IAT_Total', 'Bwd_IAT_Mean', 'Bwd_IAT_Std', 'Bwd_IAT_Max',
        'Bwd_IAT_Min', 'Fwd_PSH_Flags', 'Bwd_PSH_Flags', 'Fwd_URG_Flags', 'Bwd_URG_Flags',
        'Fwd_Header_Length', 'Bwd_Header_Length', 'Fwd_Packets_per_s', 'Bwd_Packets_per_s',
        'Packet_Length_Min', 'Packet_Length_Max', 'Packet_Length_Mean', 'Packet_Length_Std',
        'Packet_Length_Variance', 'FIN_Flag_Count', 'SYN_Flag_Count', 'RST_Flag_Count',
        'PSH_Flag_Count', 'ACK_Flag_Count', 'URG_Flag_Count', 'CWR_Flag_Count',
        'ECE_Flag_Count', 'Down_per_Up_Ratio', 'Average_Packet_Size', 'Fwd_Segment_Size_Avg',
        'Bwd_Segment_Size_Avg', 'Fwd_Bytes_per_Bulk_Avg', 'Fwd_Packet_per_Bulk_Avg',
        'Fwd_Bulk_Rate_Avg', 'Bwd_Bytes_per_Bulk_Avg', 'Bwd_Packet_per_Bulk_Avg',
        'Bwd_Bulk_Rate_Avg', 'Subflow_Fwd_Packets', 'Subflow_Fwd_Bytes',
        'Subflow_Bwd_Packets', 'Subflow_Bwd_Bytes', 'FWD_Init_Win_Bytes',
        'Bwd_Init_Win_Bytes', 'Fwd_Act_Data_Pkts', 'Fwd_Seg_Size_Min',
        'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean',
        'Idle_Std', 'Idle_Max', 'Idle_Min'
    ],
    categorical_fields=['Src_Port', 'Dst_Port', 'Protocol'],
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