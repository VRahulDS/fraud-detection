from kedro.pipeline import Pipeline, node
from .nodes import (
    merge_datasets,
    engineer_features,
    handle_missing_values,
    encode_categorical_variables,
    split_features_target
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=merge_datasets,
                inputs=["train_transaction", "train_identity"],
                outputs="merged_data",
                name="merge_tables_node"
            ),
            node(
                func=engineer_features,
                inputs=["merged_data", "params:data_processing"],
                outputs="featured_data",
                name="engineer_features_node"
            ),
            node(
                func=handle_missing_values,
                inputs=["featured_data", "params:data_processing"],
                outputs="clean_data",
                name="handle_missing_values_node"
            ),
            node(
                func=encode_categorical_variables,
                inputs=["clean_data", "params:data_processing"],
                outputs="encoded_data",
                name="encode_categorical_variables_node"
            ),
            node(
                func=split_features_target,
                inputs=["encoded_data", "params:data_processing"],
                outputs=["X_train", "y_train"],
                name="split_features_target_node"
            )
        ]
    )