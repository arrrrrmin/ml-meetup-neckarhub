#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 22:20:55 2019

@author: hh
"""

import numpy
import matplotlib.pyplot as plt

def scatter_plot(feat, bad_ix, annot=None, ti=None, xlim=None, ylim=None, color=['k','magenta'], axh=None):
    """
    Quick-and-dirty scatter plot routine for presentation of results
    """
    good_ix = numpy.logical_not(bad_ix)
    if axh is None:
        axh = plt.subplot(1,1,1);
    if xlim is not None:
        xix = (feat[:,0] >= xlim[0]) & (feat[:,0] <= xlim[1])
    else:
        xix = numpy.ones(feat[:,0].shape, dtype=bool)
    if ylim is not None:
        yix = (feat[:,1] >= ylim[0]) & (feat[:,1] <= ylim[1])
    else:
        yix = numpy.ones(feat[:,0].shape, dtype=bool)
    plotix = xix & yix
        
    plt.scatter(feat[good_ix & plotix, 0], feat[good_ix & plotix, 1], color=color[0], s=80, label='good')
    plt.scatter(feat[bad_ix & plotix, 0], feat[bad_ix & plotix, 1], color=color[1], s=80, label='bad')
    axh.set_xlim(xlim)
    axh.set_ylim(ylim)
    if annot is not None:
        for i in range(feat.shape[0]):
            if plotix[i]:
                plt.annotate(annot[i],xy=(feat[i,0], feat[i,1]), fontsize=8)
    plt.xlabel('Reduced dim 1');
    plt.ylabel('Reduced dim 2');
    if ti is not None:
        plt.title(ti)
    plt.legend();


def plot_decision_function(X, y, clf, ax):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, plot_step),
                         numpy.arange(y_min, y_max, plot_step))

    Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')
