#!/usr/bin/env python3
"""
Create Sample Training Dataset
==============================

Create a random, sample dataset with two variables, 'jet_pt' and 'etmiss'.
"""

import click

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def signal_boundary(center_x, center_y, radius):
    """Returns a matplotlib Circle object with center at (`center_x`, `center_y`)
    and radius `radius`.
    """
    return plt.Circle((center_x, center_y), radius, fill=False, ls="--", lw=2)


@click.command()
@click.option("-n", "--nevents", type=int, default=1024, help="Number of events to generate.")
@click.option("-b", "--blur", type=int, default=0, help="Blur the signal/background boundary.")
@click.option("-p", "--plot", is_flag=True, help="Generate summary plots of the newly-created dataset.")
def create_dataset(nevents, blur, plot):
    """Create a random, sample dataset with two variables, 'jet_pt' and 'etmiss'.

    Events are classified as 'signal' or 'not signal' (i.e. background). The boundary between these
    two regions is a circle in jet_pt/etmiss phase space, centred at jet_pt = 0 GeV and etmiss = 20
    GeV with a radius of 50 GeV. Events outside this circle are classified as 'signal'. It is also
    possible to 'blur' this boundary by adding a Gaussian noise term to each variable when
    classifying that event.
    """
    # Signal/background boundary parameters (circle center and radius)
    C_JET_PT = 0
    C_ETMISS = 20
    C_RADIUS = 50

    # Generate events
    rng = np.random.default_rng()
    jet_pt = rng.gamma(2, 2, nevents) * 15
    etmiss = rng.gamma(2, 2, nevents) * 20

    # Classify events
    if blur:
        blur1 = rng.normal(scale=blur, size=nevents)
        blur2 = rng.normal(scale=blur, size=nevents)
        is_signal = (jet_pt + blur1 - C_JET_PT) ** 2 + (
            etmiss + blur2 - C_ETMISS
        ) ** 2 > C_RADIUS ** 2
    else:
        is_signal = (jet_pt - C_JET_PT) ** 2 + (etmiss - C_ETMISS) ** 2 > C_RADIUS ** 2

    # Save as data frame
    df = pd.DataFrame({"jet_pt": jet_pt, "etmiss": etmiss, "is_signal": is_signal})

    foutname = "data/dataset."
    foutname += f"n{nevents}."
    foutname += f"blur{blur}." if blur else ""
    foutname += "csv"
    click.echo(f"Writing dataset to '{foutname}'")
    df.to_csv(foutname)

    # Plot
    if plot:
        signal_df = df.loc[df["is_signal"]]
        backgr_df = df.loc[~df["is_signal"]]

        fig, ax = plt.subplots()

        # Plot signal and background points
        ax.scatter(
            backgr_df["jet_pt"], backgr_df["etmiss"], c="tab:red", alpha=0.5, label="Background"
        )
        ax.scatter(
            signal_df["jet_pt"], signal_df["etmiss"], c="tab:blue", alpha=0.5, label="Signal"
        )

        # Plot "true" boundary between signal and background regions
        ax.add_artist(signal_boundary(C_JET_PT, C_ETMISS, C_RADIUS))

        ax.set_xlabel(r"jet $p_{\mathrm{T}}$ [GeV]", loc="right")
        ax.set_ylabel(r"$E_{\mathrm{T}}^{\mathrm{miss}}$ [GeV]", loc="top")

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.15)

        # Display window zoomed in on boundary region if plotting many events
        if nevents > 100:
            axins = zoomed_inset_axes(ax, zoom=2.5, loc="upper right")
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
            axins.set_xlim(0, 1.1 * (C_JET_PT + C_RADIUS))
            axins.set_ylim(0, 1.1 * (C_ETMISS + C_RADIUS))

            # Plot zoom window
            axins.scatter(backgr_df["jet_pt"], backgr_df["etmiss"], c="tab:red", alpha=0.5)
            axins.scatter(signal_df["jet_pt"], signal_df["etmiss"], c="tab:blue", alpha=0.5)
            axins.add_artist(signal_boundary(C_JET_PT, C_ETMISS, C_RADIUS))

        # Legend
        ax.legend(loc="upper left")

        # Write to file
        foutname = foutname.replace(".csv", ".pdf")
        click.echo(f"Creating plot '{foutname}'")
        fig.savefig(foutname, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    create_dataset()
