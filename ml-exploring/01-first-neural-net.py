#!/usr/bin/env python3
"""
First Neural Network
====================

Author: Joey Carter <joey.snarrcarter@gmail.com>
"""

import os

import click

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras


def create_model(n_hidden_layers, n_hidden_units):
    """Create the neural network model.
    """
    model = keras.models.Sequential()

    # Add hidden layer(s)
    for i in range(n_hidden_layers):
        model.add(keras.layers.Dense(n_hidden_units, activation="relu"))

    # Add output layer
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="rmsprop",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


@click.group()
def cli():
    """A (very simple) deep neural network to classify events.
    """
    pass


@cli.command()
@click.argument("fname", type=click.File("r"))
@click.option("-l", "--layers", type=int, default=2, help="Number of hidden layers.")
@click.option("-u", "--units", type=int, default=10, help="Number of units per hidden layer.")
@click.option("-e", "--epochs", type=int, default=5, help="Number of epochs.")
@click.option("-b", "--batch-size", type=int, default=1, help="Batch size.")
def train(fname, layers, units, epochs, batch_size):
    """Create and train the model using the given training dataset.
    """
    # Import training data and create TensorFlow dataset
    click.echo(f"Reading data from file '{fname.name}'")
    df = pd.read_csv(fname, index_col=0)
    is_signal = df.pop("is_signal")
    dataset = tf.data.Dataset.from_tensor_slices((df.values, is_signal.values))

    # Shuffle and batch the dataset
    train_dataset = dataset.shuffle(len(df)).batch(batch_size)

    # Train the model
    model = create_model(layers, units)
    model.fit(train_dataset, epochs=epochs)

    # Save
    foutname = os.path.basename(__file__).replace(".py", ".tfmodel")
    click.echo(f"Saving model to '{foutname}'")
    model.save(foutname)


@cli.command()
@click.argument("fname", type=click.File("r"))
def evaluate(fname):
    """Evaluate the trained model using the given test dataset.
    """
    # Load model
    click.echo("Loading model...")
    model = tf.keras.models.load_model(os.path.basename(__file__).replace(".py", ".tfmodel"))
    model.summary()

    # Import test data
    click.echo(f"Reading data from file '{fname.name}'")
    df = pd.read_csv(fname, index_col=0)
    is_signal = df.pop("is_signal")

    model.evaluate(df, is_signal)


@cli.command()
@click.argument("fname", type=click.File("r"))
def predict(fname):
    """Use the trained model to classify events.
    """
    # Load model
    click.echo("Loading model...")
    model = tf.keras.models.load_model(os.path.basename(__file__).replace(".py", ".tfmodel"))
    model.summary()

    # Import data
    click.echo(f"Reading data from file '{fname.name}'")
    df = pd.read_csv(fname, index_col=0)

    predictions = model.predict(df.drop(columns=["is_signal"]))

    df_with_predictions = df.assign(is_signal_dnn=predictions)

    pd.set_option("display.max_rows", 100)
    click.echo("Results:")
    click.echo(df_with_predictions)


if __name__ == "__main__":
    cli()
