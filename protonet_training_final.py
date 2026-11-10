"""Audited standard Euclidean ProtoNet objective; no classifier or extra loss."""
import numpy as np
import tensorflow as tf


def prototype_loss(support, query, support_labels, query_labels, ways=4):
    prototypes = tf.math.unsorted_segment_mean(support, support_labels, ways)
    logits = -tf.reduce_sum(tf.square(query[:, None, :] - prototypes[None, :, :]), axis=-1)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=query_labels, logits=logits))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, -1, output_type=tf.int32), query_labels), tf.float32))
    return loss, accuracy


def episode_indices(labels, allowed, classes, rng, support=5, query=10):
    """One group per input row; within-episode sampling without replacement."""
    s, q = [], []
    allowed = np.asarray(allowed, int)
    for c in classes:
        pool = allowed[labels[allowed] == c]
        if len(pool) < support + query:
            raise ValueError('Insufficient legal groups for class ' + c)
        draw = rng.choice(pool, support + query, replace=False)
        s.extend(draw[:support]); q.extend(draw[support:])
    assert not set(s) & set(q)
    return np.asarray(s, int), np.asarray(q, int)


def local_labels(k, q):
    return tf.repeat(tf.range(4, dtype=tf.int32), k), tf.repeat(tf.range(4, dtype=tf.int32), q)
