#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import MADCUBA_plotter

#++++++++++ Function Parameters ++++++++++#
"""
Description in MADCUBA_plotter.py
"""
# Figure properties
m           = 4                       
n           = 1                       
fig_name    = 'L1517B'
fig_format  ='png'
x_limits    = [0.4382877270507813, 11.149398552246094]      
y_limits    = [-0.14366640575230122, 0.6607287792116403]
x_label     = r'Vrad (km/s)'
y_label     = r'K'
drawstyle='histogram'

# Spectrum color
spec_color='k'
linewidth=0.8
linestyle='-'

# Fit colors
fit_color=['red']
fit_linewidth=0.8
fit_linestyle='-'

# Fonts
labelsize=12
labelfont='Arial'
molfontize=6
moleculefont='courier new'
panel_naming='letters'
panelfont='Arial'
panfontsize=8

# Calling function
MADCUBA_plotter.MADCUBA_plot(m, n, fig_name=fig_name, fig_format=fig_format,
                 x_limits=x_limits, y_limits=y_limits,
                 x_label=x_label, y_label=y_label, drawstyle=drawstyle, 
                 spec_color=spec_color, linewidth=linewidth, linestyle=linestyle,
                 fit_color=fit_color, fit_linewidth=fit_linewidth, fit_linestyle=fit_linestyle, 
                 labelsize=labelsize, labelfont=labelfont, molfontize=molfontize, moleculefont=moleculefont,
                 panel_naming=panel_naming, panelfont=panelfont, panfontsize=panfontsize)
