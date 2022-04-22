"""
Title: Dense Passage Retriever on TPU
Author: [Ankur Singh](https://twitter.com/ankur310794)
Date created: 2021/06/24
Last modified: 2021/06/24
Description: Implement a Dense Passage Retriever using NQ-Wikipedia Dataset.
"""
"""
## Introduction
Open-domain question answering relies on efficient passage retrieval to select 
candidate contexts, where traditional sparse vector space models, such as TF-IDF 
or BM25, are the defacto method.
We can implement using dense representations, where embeddings are learned from 
a small number of questions and passages by a simple dual-encoder framework.
Original Paper [link](https://arxiv.org/pdf/2004.04906.pdf)
"""

"""
## Setup
Install `transformers`, `faiss-cpu` via `pip install -q transformers faiss-cpu``.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFAutoModel


"""
## Model Preparation
"""


class QueryModel(tf.keras.Model):

    def __init__(self, model_config, from_pt=False, **kwargs):
        super().__init__(**kwargs)
        # Load Pretrained models
        self.query_encoder = TFAutoModel.from_pretrained(model_config.model_name, from_pt=from_pt).encoder
        self.flatten = layers.Flatten()
        self.last_dense = layers.Dense(768)
        # Add dropout layer
        self.dropout = layers.Dropout(model_config.dropout)

    def call(self, inputs, training=False, **kwargs):
        output = self.query_encoder(inputs, training=training, **kwargs)[0]
        output = self.flatten(output)
        output = self.last_dense(output, training=training)
        output = self.dropout(output, training=training)
        return output


class PassageModel(tf.keras.Model):
    """Passage Model"""

    def __init__(self, model_config, from_pt=False, **kwargs):
        super().__init__(**kwargs)
        # Load Pretrained models
        self.passage_encoder = TFAutoModel.from_pretrained(model_config.model_name, from_pt=from_pt).encoder
        self.flatten = layers.Flatten()
        self.last_dense = layers.Dense(768)
        # Add dropout layer
        self.dropout = layers.Dropout(model_config.dropout)

    def call(self, inputs, training=False, **kwargs):
        output = self.passage_encoder(inputs, training=training, **kwargs)[0]
        output = self.flatten(output)
        output = self.last_dense(output, training=training)
        output = self.dropout(output, training=training)
        return output


def cross_replica_concat(values):

    context = tf.distribute.get_replica_context()
    gathered = context.all_gather(values, axis=0)

    return tf.roll(
      gathered,
      -context.replica_id_in_sync_group * values.shape[0],
      axis=0
    )


class BiEncoderModel(tf.keras.Model):
    """Bi-Encoder Query & Passage Model"""

    def __init__(
        self,
        query_encoder,
        passage_encoder,
        num_passages_per_question,
        model_config,
        strategy,
        global_batch_size,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Query encoder model
        self.query_encoder = query_encoder
        # Passage encoder model
        self.passage_encoder = passage_encoder
        # No. positives plus No. of hard negatives
        self.num_passages_per_question = num_passages_per_question
        # Model configuration
        self.model_config = model_config
        # Execution strategy (colab)
        self.strategy = strategy
        # Loss tracker
        self.loss_tracker = keras.metrics.Mean(name="loss")
        # Define loss
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(
            reduction=keras.losses.Reduction.NONE, from_logits=True
        )

        self.global_batch_size = global_batch_size

    def calculate_loss(self, logits):
        """Function to calculate in batch loss"""

        # Make In-Batch Labels:
        # Given single quetion positives are placed first followed by negatives.
        labels = tf.convert_to_tensor(
            [
                i
                for i in range(
                    0,
                    (self.global_batch_size * self.num_passages_per_question),
                    self.num_passages_per_question,
                )
            ]
        )

        loss = self.loss_fn(labels, logits)
        scale_loss = tf.reduce_sum(loss) * (1.0 / self.global_batch_size)
        return scale_loss

    def passage_forward(self, X):

        # Reshape input (BS, num_passages_per_question, seq_len) -> (BS*num_passages_per_question, seq_len)
        input_shape = (
            self.model_config.batch_size_per_replica * self.num_passages_per_question,
            self.model_config.passage_max_seq_len,
        )
        input_ids = tf.reshape(X["passage_input_ids"], input_shape)
        attention_mask = tf.reshape(X["passage_attention_mask"], input_shape)
        token_type_ids = tf.reshape(X["passage_token_type_ids"], input_shape)
        
        input_ids = tf.cast(input_ids, tf.int32)
        attention_mask = tf.cast(attention_mask, tf.int32)
        token_type_ids = tf.cast(token_type_ids, tf.int32)

        # Call passage encoder model
        outputs = self.passage_encoder(
            [input_ids, attention_mask, token_type_ids], training=True
        )
        return outputs

    def query_forward(self, X):
        # Reshape input (BS, seq_len) -> (BS, seq_len)
        input_shape = (
            self.model_config.batch_size_per_replica,
            self.model_config.query_max_seq_len,
        )
        input_ids = tf.reshape(X["query_input_ids"], input_shape)
        attention_mask = tf.reshape(X["query_attention_mask"], input_shape)
        token_type_ids = tf.reshape(X["query_token_type_ids"], input_shape)

        input_ids = tf.cast(input_ids, tf.int32)
        attention_mask = tf.cast(attention_mask, tf.int32)
        token_type_ids = tf.cast(token_type_ids, tf.int32)
        
        outputs = self.query_encoder(
            [input_ids, attention_mask, token_type_ids], training=True
        )
        return outputs

    def train_step(self, X):

        with tf.GradientTape() as tape:
            # Call encoder models
            passage_embeddings = self.passage_forward(X)
            query_embeddings = self.query_forward(X)

            # Get all replica concat values for In-Batch loss calculation
            global_passage_embeddings = cross_replica_concat(passage_embeddings)
            global_query_embeddings = cross_replica_concat(query_embeddings)

            # Dot product similarity
            similarity_scores = tf.linalg.matmul(
                global_query_embeddings, global_passage_embeddings, transpose_b=True
            )

            loss = self.calculate_loss(similarity_scores)
            loss = loss / self.strategy.num_replicas_in_sync

        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, X):
        # Call encoder models
        passage_embeddings = self.passage_forward(X)
        query_embeddings = self.query_forward(X)

        # Get all replica concat values for In-Batch loss calculation
        global_passage_embeddings = cross_replica_concat(passage_embeddings)
        global_query_embeddings = cross_replica_concat(query_embeddings)

        # Dot product similarity
        similarity_scores = tf.linalg.matmul(
            global_query_embeddings, global_passage_embeddings, transpose_b=True
        )

        loss = self.calculate_loss(similarity_scores)
        loss = loss / self.strategy.num_replicas_in_sync

        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]
