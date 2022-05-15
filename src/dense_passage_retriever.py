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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import TFAutoModel


"""
## Model Preparation
"""


class QueryModel(tf.keras.Model):
    """Query Model"""

    def __init__(self, model_config, from_pt=False, **kwargs):
        super().__init__(**kwargs)
        # Load Pretrained models
        self.query_encoder = TFAutoModel.from_pretrained(model_config.model_name, from_pt=from_pt)
        # Add dropout layer
        self.dropout = layers.Dropout(model_config.dropout)

    def call(self, inputs, training=False, **kwargs):

        pooled_output = self.query_encoder(inputs, training=training, **kwargs)[0][:, 0]
        pooled_output = self.dropout(pooled_output, training=training)
        return pooled_output


class PassageModel(tf.keras.Model):
    """Passage Model"""

    def __init__(self, model_config, from_pt=False, **kwargs):
        super().__init__(**kwargs)
        # Load Pretrained models
        self.passage_encoder = TFAutoModel.from_pretrained(model_config.model_name, from_pt=from_pt)
        # Add dropout layer
        self.dropout = layers.Dropout(model_config.dropout)

    def call(self, inputs, training=False, **kwargs):

        pooled_output = self.passage_encoder(inputs, training=training, **kwargs)[0][:,0]
        pooled_output = self.dropout(pooled_output, training=training)
        return pooled_output


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
        num_questions_per_passage,
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
        self.num_questions_per_passage = num_questions_per_passage
        # Model configuration
        self.model_config = model_config
        # Execution strategy (colab)
        self.strategy = strategy
        # Loss tracker
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_typo_tracker = keras.metrics.Mean(name="loss_typo")
        # Define loss
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(
            reduction=keras.losses.Reduction.NONE, from_logits=True
        )
        self.loss_fn2 = keras.losses.CategoricalCrossentropy(
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

    def calculate_loss2(self, logits):
        """Function to calculate in batch loss"""
        # Make In-Batch Labels:
        # Given single quetion positives are placed first followed by negatives.

        labels = [[0 for _ in range(self.global_batch_size * self.num_questions_per_passage)] for _ in range(self.global_batch_size)]
        for i in range(self.global_batch_size):
            labels[i][i * 2] = 1
            labels[i][i * 2 + 1] = 0.5
        labels = tf.convert_to_tensor(labels)

        loss = self.loss_fn2(labels, logits)
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
        token_type_ids = tf.zeros(shape=input_shape, dtype=tf.int32)

        input_ids = tf.cast(input_ids, tf.int32)
        attention_mask = tf.cast(attention_mask, tf.int32)

        # Call passage encoder model
        outputs = self.passage_encoder(
            [input_ids, attention_mask, token_type_ids], training=True
        )
        return outputs

    def query_forward(self, X, prefix="query"):
        # Reshape input (BS, seq_len) -> (BS, seq_len)
        input_shape = (
            self.model_config.batch_size_per_replica,
            self.model_config.query_max_seq_len,
        )
        input_ids = tf.reshape(X[f"{prefix}_input_ids"], input_shape)
        attention_mask = tf.reshape(X[f"{prefix}_attention_mask"], input_shape)
        token_type_ids = tf.zeros(shape=input_shape, dtype=tf.int32)

        input_ids = tf.cast(input_ids, tf.int32)
        attention_mask = tf.cast(attention_mask, tf.int32)
        
        outputs = self.query_encoder(
            [input_ids, attention_mask, token_type_ids], training=True
        )
        return outputs

    def train_step(self, X):

        with tf.GradientTape() as tape:
            # Call encoder models
            passage_embeddings = self.passage_forward(X)
            query_embeddings = self.query_forward(X, prefix="query")
            partialcredit_embeddings = self.query_forward(X, prefix="partialcredit")
            typo_embeddings = self.query_forward(X, prefix="typo")

            # Get all replica concat values for In-Batch loss calculation
            global_passage_embeddings = cross_replica_concat(passage_embeddings)
            global_query_embeddings = cross_replica_concat(query_embeddings)
            global_partialcredit_embeddings = cross_replica_concat(partialcredit_embeddings)
            global_typo_embeddings = cross_replica_concat(typo_embeddings)

            # Interleave partials
            total_len = global_query_embeddings.shape[0] + global_partialcredit_embeddings.shape[0]
            combined_embeddings = tf.dynamic_stitch([range(0, total_len, 2), range(1, total_len + 1, 2)],
                                                    [global_query_embeddings, global_partialcredit_embeddings])

            # Dot product similarity
            similarity_scores = tf.linalg.matmul(
                combined_embeddings, global_passage_embeddings, transpose_b=True
            )

            typo_scores = tf.linalg.matmul(
                global_query_embeddings, global_typo_embeddings, transpose_b=True
            )

            loss_orig = self.calculate_loss2(similarity_scores) / self.strategy.num_replicas_in_sync
            loss_typo = self.calculate_loss(typo_scores) / self.strategy.num_replicas_in_sync
            loss = loss_orig + loss_typo

        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Monitor loss
        self.loss_tracker.update_state(loss)
        self.loss_typo_tracker.update_state(loss_typo)
        return {"loss": self.loss_tracker.result(), "loss_typo": self.loss_typo_tracker.result()}

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
        return [self.loss_tracker, self.loss_typo_tracker]
