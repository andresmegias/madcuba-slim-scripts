#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MADCUBA Lines Plotter
---------------------
Version 1.9

Copyright (C) 2023 - Andrés Megías Toledano

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

config_file = 'examples/MgC4H.yaml'

# Libraries and functions.

import os
import re
import sys
import copy
import platform
import itertools
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

ssf = 30  # supersampling factor

def gaussian(x, amplitude=1, mean=0, std=1, subs=ssf):
    """
    Apply a subsampled gaussian function to the input array.

    Parameters
    ----------
    x : array
        Input array.
    amplitude : float, optional
        Maximum value of the curve. The default is 1.
    mean : float, optional
        Mean of the curve. The default is 0.
    std : float, optional
        Standard deviation of the curve. The default is 1.
    subs : int, optional
        Number of bins in which subsample the gaussian function.

    Returns
    -------
    y : array
        Output array.
    """
    dx = x[1] - x[0]
    x2 = np.linspace(x[0]-dx/2, x[-1]+dx/2, subs*len(x))
    y2 = amplitude * np.exp(-(x2-mean)**2 / (2*std**2))
    y = y2.reshape(-1,subs).mean(axis=1)
    return y

def multigaussian(x, *params):
    """
    Apply several gaussian functions to the input array.

    Parameters
    ----------
    x : array
        Input array.
    *params : tuple
        Tuple of the parameters of the gaussian functions. For each function,
        there must be 3 parameters: amplitude, mean and standard deviation.

    Returns
    -------
    y : array
        Output array.
    """
    y = np.zeros(len(x))
    for i in range(0, len(params), 3):
        height = params[i]
        mean = params[i+1]
        std = params[i+2]
        y += gaussian(x, height, mean, std)
    return y

def multigaussian_fit(x, y, num_curves=1, max_iters=20, verbose=False,
                      old_results=None):
    """
    Make a fit of multiple gaussians to the input data.

    Parameters
    ----------
    x : array
        Independent variable.
    y : array
        Dependent variable.
    num_curves : int, optional
        Starting number of curves to fit. The default is 1.
    max_iters : int, optional
        Number of maximum calls of the function. The default is 20.
    verbose : bool, optional
        If True, return informative messages of the fit.

    Returns
    -------
    popt: list
        List of the parameters of the curves fitted, in groups of 3.
    width: float
        Approximate width of the combined curves.
    r2: float
        Coefficient of determination of the fit.
    """
    # First guess of the parameters of the gaussians.    
    height = np.mean(y)
    mean = np.mean(x)
    std = np.std(x) / 2
    means = np.random.normal(mean, std, num_curves)
    guess = []
    for i in range(len(means)):
        guess += [height, means[i], std]
    # Fit.
    try:
        popt, pcov = curve_fit(multigaussian, x, y, p0=guess)
    except RuntimeError:
        if verbose:
            print('Fitting failed, trying again. (RuntimeError)')
            print(num_curves, '-', max_iters)
        if max_iters > 0:
            return multigaussian_fit(x, y, num_curves, max_iters-1, verbose)
        else:
            if old_results:
                return old_results
            else:
                return [], None, None
    # Coefficient of determination.    
    res = y - multigaussian(x, *popt)
    ss_res = np.sum(res**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - (ss_res / ss_tot)
    if verbose:
        print(num_curves, r2, max_iters)
    # Height of the fit.
    heights = popt[0::3]
    mse = np.mean(res**2)
    if (y.max() - y.min())**2 < mse and min(heights) < 2*mse:
        if verbose: 
            print('Fitting failed, trying again. (too much noise)')
            print(num_curves, '-', max_iters)
            print(y.min(), y.max(), mse)
        if max_iters > 0:
            return multigaussian_fit(x, y, num_curves, max_iters-1, verbose)
        else:
            return [], None, None
    # Window of the fit.
    means = popt[1::3]
    sigmas = abs(popt[2::3])
    i1 = means.argmin()
    i2 = means.argmax()
    window = [means[i1] - 4*sigmas[i1], means[i2] + 4*sigmas[i2]]
    # Condition for finishing or adding another gaussian curve.
    if old_results and old_results[-1] > 0.5 and (r2 - old_results[-1]) < 0.05:
        if verbose:
            print('Returning to previous fit.')
        if old_results[-1] > 0.9:
            if verbose:
                print('Finished.')
            return tuple(old_results)
        else:
            return [], None, None
    if r2 > 0.9 or num_curves > 5 or max_iters < 1:
        if r2 > 0.9:
            if verbose:
                print('Finished.')
            return popt, window, r2
        else:
            return [], None, None
    else: 
        num_curves += 1
        old_popt = copy.copy(popt)
        old_window = copy.copy(window)
        old_r2 = copy.copy(r2)
        old_results = [old_popt, old_window, old_r2]
        return multigaussian_fit(x, y, num_curves, max_iters-1, verbose,
                                 old_results)

def format_species_name(input_name, simplify_numbers=True, acronyms={}):
    """
    Format text as a molecule name, with subscripts and upperscripts.

    Parameters
    ----------
    input_name : str
        Input text.
    simplify_numbers : bool, optional
        Remove the brackets between single numbers.
    acronyms : dict
        Dictionary of acronyms for species name. If the input text is one of
        the dictionary keys, it will be replaced by the corresponding value,
        and then the formatting function will be still applied.

    Returns
    -------
    output_name : str
        Formatted molecule name.
    """
    original_name = copy.copy(input_name)
    # acronyms
    for name in acronyms:
        if original_name == name:
            original_name = acronyms[name]
    # removing the additional information of the transition
    original_name = original_name.replace('_k',',k').split(',')[0]
    # prefixes
    possible_prefixes = ['#', '@', '$']
    if '-' in original_name and original_name.split('-')[0].isalpha():
        prefix = original_name.split('-')[0]
    else:
        prefix = ''
        for text in possible_prefixes:
            if original_name.startswith(text):
                prefix = text
                break
    original_name = original_name.replace(prefix, '')
    # upperscripts
    possible_upperscript, in_upperscript = False, False
    output_name = ''
    upperscript = ''
    inds = []
    for (i, char) in enumerate(original_name):
        if (char.isupper() and not possible_upperscript
                and '-' in original_name[i:]):
            inds += [i]
            possible_upperscript = True
        elif char.isupper():
            inds += [i]
        if char == '-' and not in_upperscript:
            inds += [i]
            in_upperscript = True
        if in_upperscript and not (char.isdigit() or char == '-'):
            in_upperscript = False
        if not possible_upperscript:
            output_name += char
        if in_upperscript and i != inds[-1]:
            if char.isdigit():
                upperscript += char
            if char == '-' or i+1 == len(original_name):
                if len(inds) > 2:
                    output_name += original_name[inds[0]:inds[-2]]
                    output_name += ('$^{' + upperscript + '}$'
                                    + original_name[inds[-2]:inds[-1]])
                upperscript = ''
                in_upperscript, possible_upperscript = False, False
                inds = []
    if output_name == '':
        output_name = original_name
    output_name = output_name.replace('[', '^{').replace(']', '}')
    if output_name.endswith('+') or output_name.endswith('-'):
        symbol = output_name[-1]
        output_name = output_name.replace(symbol, '$^{'+symbol+'}$')
    original_name = copy.copy(output_name)
    # subscripts
    original_name = original_name.replace('_','')
    output_name, subscript, prev_char = '', '', ''
    in_bracket = False
    for (i, char) in enumerate(original_name):
        if char == '{':
            in_bracket = True
        elif char == '}':
            in_bracket = False
        if (char.isdigit() and not in_bracket
                and prev_char not in ['=', '-', '{', ',']):
            subscript += char
        else:
            if len(subscript) > 0:
                output_name += '$_{' + subscript + '}$'
                subscript = ''
            output_name += char
        if i+1 == len(original_name) and len(subscript) > 0:
            output_name += '$_{' + subscript + '}$'
        prev_char = char
    output_name = output_name.replace('^$_', '$^').replace('$$', '')
    # some formatting
    output_name = output_name.replace('$$', '').replace('__', '')
    # remove brackets from single numbers
    if simplify_numbers:
        single_numbers = re.findall('{(.?)}', output_name)
        for number in set(single_numbers):
            output_name = output_name.replace('{'+number+'}', number)
    # prefix
    if prefix == '$':
        prefix = '\$'
    if prefix in ('\$', '#', '@'):
        prefix += '$\,$'
    output_name = prefix + output_name
    output_name = output_name.replace('$^$', '')
    return output_name

def format_quantum_numbers(input_numbers):
    """
    Format text as a molecular transition.

    Parameters
    ----------
    input_numbers : str
        Input text with the transition quantum numbers separated with commas
        (,) and hyphens (-).

    Returns
    -------
    output_numbers : str
        Formatted quantum numbers.
    """
    if input_numbers.count('-') == 1:
        input_numbers = input_numbers.replace('-', ';')
    if not ';' in input_numbers:
        sep = ',' if ',' in input_numbers else '-'
        numbers = input_numbers.split(sep)
        num_qn = len(numbers) // 2
        input_numbers = ','.join(numbers[:num_qn]) + ';' + \
            ','.join(numbers[num_qn:])
    if ';' in input_numbers:
        numbers1, numbers2 = input_numbers.split(';')
        first_number = numbers1.split(',')[0]
        subindices = ','.join(numbers1.split(',')[1:])
        numbers1 = '$' + first_number + '_{' + subindices + '}$'
        first_number = numbers2.split(',')[0]
        subindices = ','.join(numbers2.split(',')[1:])
        numbers2 = '$' + first_number + '_{' + subindices + '}$'
        output_numbers = '>'.join([numbers1, numbers2])
        output_numbers = output_numbers.replace('>', ' $\\rightarrow$ ')
        output_numbers = output_numbers.replace('_{}', '')
    else:
        output_numbers = input_numbers
    return output_numbers

def check_overlap(positions, plot_interval=None, width=0.01):
    """
    Check if any of the given label positions overlap.
    
    Parameters
    ----------
    positions : list (float)
        Positions to check.
    plot_interval : float, optional
        Horizontal range of the plot in which the labels will appear.
    width : float, optional
        Minimum separation betweeen positions to check the overlaping.
        The default is 0.01.

    Returns
    -------
    overlap : bool.
        If True, there is overlaping of at least two positions.
    overlap_mask : list (int)
        Indices of the positions which overlap.
    overlap_group_inds : list (list)
        List of the groups containing each line that overap in that group.
    """
    positions = np.array(positions)
    num_positions = len(positions)
    if plot_interval is None:
        plot_interval = [positions.min(), positions.max()]
    plot_range = plot_interval[1] - plot_interval[0]
    combinations_inds = \
        list(itertools.combinations(np.arange(num_positions), 2))
    overlap = False
    overlap_mask = [False] * num_positions
    for pair in combinations_inds:
        i1, i2 = pair
        x1, x2 = positions[i1], positions[i2]
        if abs(x2-x1) < width*plot_range:
            overlap = True
            overlap_mask[i1] = True
            overlap_mask[i2] = True
    overlap_group_inds = []
    overlap_inds = []
    new_group = False
    group = []
    sorted_inds = np.argsort(positions)
    last_idx = sorted_inds[-1]
    for i in sorted_inds: 
        if overlap_mask[i]:
            new_group = True
            group += [i]
            overlap_inds += [i]
        if ((not overlap_mask[i] and new_group)
             or (overlap_mask[i] and i == last_idx)):
                overlap_group_inds += [group]
                group = []
                new_group = False
    return overlap, overlap_mask, overlap_group_inds


def create_label_positions(initial_positions, plot_interval, width=0.01,
                           max_iters=3000):
    """
    Generates label positions that not overlap from the given positions.

    Parameters
    ----------
    initial_positions : list (float)
        Initial positions.
    plot_interval : float
        Horizontal range of the plot in which the labels will appear.
    width : float, optional
        Minimum separation betweeen labels to check the overlaping.
        The default is 0.01.
    max_iters : int, optional
        Maximum number of iterations for finding the proper positions.

    Returns
    -------
    new positions : list (float).
        Generated positions, that not overlap.
    """
    plot_range = plot_interval[1] - plot_interval[0]
    overlap, overlap_mask, _, = check_overlap(initial_positions, plot_interval,
                                              width)
    if not overlap:
        return initial_positions
    else:
        for i in range(max_iters):
            factor = 0.3*(i+1)/1000
            new_positions = []
            for (j, x) in enumerate(initial_positions):
                if overlap_mask[j]:
                    xx = np.random.normal(x, factor*width*plot_range)
                    xx = max(plot_interval[0], xx)
                    xx = min(xx, plot_interval[1])
                else:
                    xx = x
                new_positions += [xx]
            overlap, overlap_mask, _, = \
                check_overlap(new_positions, plot_interval, width)
            if not overlap:
                break
        if overlap:
            x1, x2 = min(new_positions), max(new_positions)
            new_positions = np.linspace(x1, x2, len(positions)).tolist()
        return new_positions

def parse_text_location(text, fig):
    """
    Return the coordinates of the input annotation text.

    Parameters
    ----------
    text : Text (matplotlib)
        Text object from Matplotlib.
    fig : Figure (matplotlib)
        Figure object, from Matpotlib, where the text annotation was made.

    Returns
    -------
    x : float
        Horizontal coorinate of the text.
    y : float
        Vertical coordinate of the text.
    """
    text = str(text.get_window_extent(renderer=fig.canvas.get_renderer()))
    text = text.replace('Bbox(','').replace(')', '')
    text = text.replace('x0=','').replace(' y0=','')
    text = text.replace(' x1=','').replace(' y1=','')
    x1, y1, x2, y2 = text.split(',')
    x1, x2 = float(x1), float(x2)
    y1, y2 = float(y1), float(y2)
    return x1, x2, y1, y2

def format_text_with_species_name(text, delim='*'):
    """Format text formatting the chemical names between the delimiter."""
    if text.count(delim) == 2:
        species_name = text.split(delim)[1]
        text = text.replace('{}{}{}'.format(delim, species_name, delim),
          format_species_name(species_name, acronyms=acronyms))
    return text

#%%

# Default options.
default_options = {
    'figure size': 'auto',
    'subplots horizontal spacing': 0.10,
    'subplots vertical spacing': 0.25,
    'font size': 10,
    'label font size': None,
    'frame width': 0.8,
    'plot line width': 1.2,
    'fit line width': None,
    'rows': 1,
    'columns': 1,
    'title height': 0.92,
    'subplot titles height': 0.90,
    'ticks direction': 'in',
    'use frequencies': False,
    'velocity limits': [],
    'frequency limits': [],
    'intensity limits': [],
    'input spectral variable': 'velocity',
    'shown spectral variable': None,
    'velocity offset': 0.,
    'velocity label': 'velocity (km/s)',
    'frequency label': 'frequency (GHz)',
    'intensity label': 'intensity (K)',
    'use common labels': False,
    'spectral factor': 1.,
    'intensity factor': 1.,
    'fit plot style': 'steps',
    'gaussian fit': False,
    'fit color': 'tab:red',
    'all species fit color': 'tab:blue',
    'other species color': 'tab:blue',
    'fill spectrum': False,
    'figure titles': [],
    'data folders': [],
    'species': [],
    'species acronyms': [],
    'additional fits': [],
    'join subplots': True,
    'show transitions': False,
    'show main species transitions': True,
    'show rest species transitions': False,
    'mark transitions with lines': True,
    'show species names': True,
    'show quantum numbers': False,
    'show subplot titles': True,
    'save figure': True,
    'show all species fit': False,
    'transitions threshold': 0.0,
    'transition labels minimum distance': 0.0
    }
gray = tuple([0.6]*3)
light_gray = tuple([0.8]*3)
plt.close('all')

# Folder sepatator.
separator = '/'
operating_system = platform.system()
if operating_system == 'Windows':
    separator = '\\'

# Configuration file.
original_folder = os.getcwd() + separator
if len(sys.argv) != 1:
    config_file = sys.argv[1]
config_path = original_folder + config_file 
config_folder = separator.join(config_path.split(separator)[:-1]) + separator
os.chdir(config_folder)
if os.path.isfile(config_path):
    with open(config_path) as file:
        config_or = yaml.safe_load(file)
else:
    raise FileNotFoundError('Configuration file not found.')
config = {**default_options, **config_or}
    
# Parameters.
figure_size = config['figure size']
font_size = config['font size']
label_font_size = config['label font size']
frame_width = config['frame width']
line_width = config['plot line width']
fit_line_width = config['fit line width']
fit_line_width = line_width if type(fit_line_width) is None else fit_line_width
num_rows = config['rows']
num_cols = config['columns']
title_height = config['title height']
subtitles_height = config['subplot titles height']
join_subplots = config['join subplots']
ticks_direction = config['ticks direction']
input_spectral_variable = config['input spectral variable']
shown_spectral_variable = config['shown spectral variable']
if shown_spectral_variable is None:
    shown_spectral_variable = input_spectral_variable
velocity_offset = config['velocity offset']
velocity_lims = config['velocity limits']
if velocity_lims != [] and type(velocity_lims[0]) in (float, int):
    velocity_lims = [velocity_lims] * num_cols
elif len(velocity_lims) != num_cols:
    velocity_lims = ['auto'] * num_cols
frequency_lims = config['frequency limits']
if frequency_lims != [] and type(frequency_lims[0]) in (float, int):
    frequency_lims = ['auto'] * num_cols
elif len(frequency_lims) != num_cols:
    frequency_lims = ['auto'] * num_cols
intensity_lims = config['intensity limits']
if intensity_lims != [] and type(intensity_lims[0]) in (float, int):
    intensity_lims = [intensity_lims] * num_rows
elif len(intensity_lims) != num_rows:
    intensity_lims = ['auto'] * num_rows
lines_lim = config['transitions threshold']
velocity_label = config['velocity label']
frequency_label = config['frequency label']
intensity_label = config['intensity label']
use_common_labels = config['use common labels']
spectral_factor = float(config['spectral factor'])
intensity_factor = float(config['intensity factor'])
if spectral_factor == 1e3:
    if 'velocity label' not in config_or:
        velocity_label = velocity_label.replace('(km/s)', '(m/s)')
    if 'frequency label' not in config_or:
        frequency_label = frequency_label.replace('(GHz)', '(MHz)')
if 'intensity label' not in config_or and intensity_factor == 1e3:
    intensity_label = intensity_label.replace('(K)', '(mK)')
fit_style = config['fit plot style']
gaussian_fit = config['gaussian fit']
usual_fit_color = config['fit color']
all_species_fit_color = config['all species fit color']
other_species_color = config['other species color']
fill_spectrum = config['fill spectrum']
show_transitions = config['show transitions']
show_main_species_lines = config['show main species transitions']
show_rest_species_lines = config['show rest species transitions']
mark_lines = config['mark transitions with lines']
show_species_names = config['show species names']
show_quantum_numbers = config['show quantum numbers']
show_subplot_titles = config['show subplot titles']
save_figure = config['save figure']
label_min_dist = config['transition labels minimum distance']
show_all_species_fit = config['show all species fit']
acronyms = config['species acronyms']
extra_fits = config['additional fits']
if join_subplots:
    ticks_direction = 'in'
multiplot = True if 'data folders' in config else False
if multiplot:
    folders = config['data folders']
    figure_titles = config['figure titles']
else:
    folders = [config['data folder']]
    figure_titles = [config['figure title']]
data_subfolder = separator + 'data' + separator
spectroscopy_subfolder = separator + 'spectroscopy' + separator
use_frequency = True if shown_spectral_variable == 'frequency' else False
if (('frequency limits' in config_or or 'frequency label' in config_or)
        and ('velocity limits' not in config_or
             and 'velocity label' not in config_or)):
    use_frequency = True
if use_frequency:
    join_subplots = False
    spectral_label = frequency_label
    spectral_lims = frequency_lims
else:
    spectral_label = velocity_label
    spectral_lims = velocity_lims
if label_font_size is None:
    label_font_size = 0.9*font_size
if show_rest_species_lines or show_quantum_numbers:
    label_font_size = min(0.7*font_size, label_font_size)
    if show_rest_species_lines and show_quantum_numbers:
        label_font_size = min(0.4*font_size, label_font_size)
else:
    label_font_size = 0.9*font_size
speed_light = 299792.458  # km/s
wspace = config['subplots horizontal spacing']
hspace = config['subplots vertical spacing']

# Graphical options.
lw = line_width
flw = fit_line_width
plt.rcParams['font.size'] = font_size
plt.rcParams['axes.linewidth'] = frame_width
for i in ['x','y']:
    plt.rcParams[i+'tick.major.width'] = frame_width
    plt.rcParams[i+'tick.minor.width'] = 0.8*frame_width
plt.rcParams['xtick.direction'] = ticks_direction
plt.rcParams['ytick.direction'] = ticks_direction
plt.rcParams['xtick.major.size'] = 5.
plt.rcParams['ytick.major.size'] = 5.
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['axes.formatter.useoffset'] = False
xpad = 7. if join_subplots else 10.
ypad = 7. if join_subplots else 10.

#%%

print()
print('MADCUBA Lines Plotter')
print('---------------------')
print()
    
for (f, folder) in enumerate(folders):
    
    if not folder.endswith(separator):
        folder += separator
        
    # Data files.
    spectrum_files, titles = [], []
    plot_fits, fit_colors, manual_fits, manual_lines = [], [], [], []
    exceptions_indices_spec, exceptions_indices_int = [], []
    exceptions_limits_spec, exceptions_limits_int = [], []
    for i, molecule in enumerate(config['species']):
        molecule = list(molecule.keys())[0]
        all_spectra = Path(folder + molecule + data_subfolder).glob('**/*')
        all_spectra = sorted([str(pp) for pp in all_spectra])
        if len(all_spectra) == 0:
            raise Exception('No files for molecule {}.'.format(molecule))
        file_prefix = config['species'][i][molecule]['file']
        for spectrum in all_spectra:
            spectrum = spectrum.split(separator)[-1]
            if type(file_prefix) in (list, tuple):
                file_prefix = file_prefix[f]
            was_file_found = False
            if spectrum.startswith(file_prefix):
                spectrum_files += [folder + molecule + data_subfolder + spectrum]
                if 'title' in config['species'][i][molecule]:
                    title = config['species'][i][molecule]['title']
                    title = title.replace('\n ','\n')
                    title = format_text_with_species_name(title)
                else:
                    title = 'auto'
                titles += [title]
                if 'fit' in config['species'][i][molecule]:
                    config_fit = config['species'][i][molecule]['fit']
                else:
                    config_fit = False
                if type(config_fit) in (list,tuple):
                    config_fit = config_fit[f]
                plot_fits += [config_fit]
                if 'lines of the fit' in config['species'][i][molecule]:
                    manual_fit = True
                else:
                    manual_fit = False
                manual_fits += [manual_fit]
                if manual_fit:
                    manual_lines_i = \
                        config['species'][i][molecule]['lines of the fit']
                    if type(manual_lines_i) in [list,tuple]:
                        manual_lines_i = manual_lines_i[f]
                    elif (type(manual_lines_i) in [list,tuple]
                          and len(manual_lines_i) == 1):
                        manual_lines_i = manual_lines_i[0]
                else:
                    manual_lines_i = {}
                manual_lines += [manual_lines_i]
                was_file_found = True
                break
        if not was_file_found:
            raise Exception('File not found for molecule {}.'
                            .format(molecule))
                
        if ('intensity limits' in config['species'][i][molecule]
                and (join_subplots and (i+1 - num_cols) % num_cols == 0
                     or not join_subplots)):
            exceptions_indices_int += [i]
            exceptions_limits_int += \
                [config['species'][i][molecule]['intensity limits']]
        if ('velocity limits' in config['species'][i][molecule]
                and (join_subplots and i < num_cols or not join_subplots)):
            exceptions_indices_spec += [i]
            exceptions_limits_spec += \
                [config['species'][i][molecule]['velocity limits']]
        if 'fit color' in config['species'][i][molecule]:
            config_color = config['species'][i][molecule]['fit color']
            if type(config_color) in (list, tuple):
                config_color = config_color[f]
            fit_colors += [config_color]
        else:
            fit_colors += [usual_fit_color]
    
    # Loading of the spectra.
    spectra, transitions_main, transitions_rest = [], [], []
    rest_freqs = []
    for file in spectrum_files:
        spectra += [np.loadtxt(file)]
        rest_freqs += [float(file.split('_')[-1]) / 1e9]
        if show_transitions:
            file = file.replace(data_subfolder, spectroscopy_subfolder)
            transitions_main += [np.loadtxt(file, str)]
            prefix = file.split(spectroscopy_subfolder)[0] + spectroscopy_subfolder
            texts = file.split(spectroscopy_subfolder)[1].split('_')
            file = '_'.join([texts[0], 'TRANSITIONS_ALL', *texts[2:]])
            file = prefix + file
            try:
                data = np.loadtxt(file, str, usecols=[0,1,2,3,5,6,7,8],
                                  delimiter='\t')
            except:
                data = np.loadtxt(file, str)
            transitions_rest += [data]
    spectra = spectra[:num_cols*num_rows]
    num_rows = len(spectra) // num_cols + int(len(spectra) % num_cols != 0)
    intensity_lims = intensity_lims[:num_rows]
    for (i, transition_group) in enumerate(transitions_main):
        transitions_main[i] = list(transition_group)
        if len(transition_group.shape) == 1:
            transition = transition_group
            transitions_main[i] = list(transition)
            if len(transition) == 3:
                transitions_main[i].insert(2, '')
        else:
            for (j, transition) in enumerate(transition_group):
                transitions_main[i][j] = list(transition)
                if len(transition) == 3:
                    transitions_main[i][j].insert(2, '')
    for (i, transition_group) in enumerate(transitions_rest):
        transitions_rest[i] = list(transition_group)
        if len(transition_group.shape) == 1:
            transition = transition_group
            transitions_rest[i] = list(transition)
            if len(transition) == 3:
                transitions_rest[i].insert(2, '')
        else:
            for (j, transition) in enumerate(transition_group):
                transitions_rest[i][j] = list(transition)
                if len(transition) == 3:
                    transitions_rest[i][j].insert(2, '')
    if transitions_main == []:
        show_main_species_lines = False
    if transitions_rest == []:
        show_rest_species_lines = False
    if not (show_main_species_lines or show_rest_species_lines):
        show_transitions = False
    
    #%% Plot.
    
    # Figure.
    if figure_size == 'auto':
        figure_size = (4.0*num_cols, 3.0*num_rows)
    fig = plt.figure(1+f, figsize=figure_size)
    if join_subplots:
        wspace, hspace = 0., 0.
    plt.subplots_adjust(left=0.13, bottom=0.11, wspace=wspace, hspace=hspace)
    # Indices of rows and columns.
    y_idx = np.arange(0, len(spectra), num_cols)
    x_idx = np.arange(len(spectra))
    x_idx = x_idx[x_idx+1 > len(spectra) - num_cols]
    
    # Plot of the data.
    axes, lines = [], []
    num_row_i = 1
    for (i, spectrum) in enumerate(spectra):
        spectralvar = spectrum[:,0]
        intensity = spectrum[:,1]
        try:
            intensity_fit = spectrum[:,2]
        except:
            plot_fits[i] = False
        if show_all_species_fit:
            intensity_fit_all = spectra[i][:,3]
        spectralvar_fit = spectralvar.copy()
        entry = list(config['species'][i].keys())[0]
        config_entry = config['species'][i][entry]
        if plot_fits[i] and fit_style == 'curve':
            N = spectrum.shape[0]
            spectralvar_fit = np.linspace(spectralvar[0], spectralvar[-1], ssf*N)
            if not manual_fits[i]:
                if gaussian_fit:
                    # guess = [intensity_fit.max(), spectralvar[N//2], 0.3]
                    # params, _ = curve_fit(gaussian, spectralvar, intensity_fit
                    #                       p0=guess)
                    params, _, _ = multigaussian_fit(spectralvar,
                                                spectralvar_fit, verbose=False)
                    intensity_fit = multigaussian(spectralvar_fit, *params)
                else:
                    interpolation = interp1d(spectralvar, intensity_fit,
                                             kind='quadratic')
                    intensity_fit = interpolation(spectralvar_fit)
            else:
                widths = np.array(manual_lines[i]['width'], float) / 2.355
                heights = np.array(manual_lines[i]['intensity'], float)
                if 'position' in manual_lines[i]:
                    means = np.array(manual_lines[i]['position'], float)
                else:
                    means = transitions_main[i][0]
                    if type(means) in (list, tuple):
                        means = means[0]
                    means = float(means)
                widths = [widths] if type(widths) is not list else widths
                heights = [heights] if type(heights) is not list else heights
                means = [means] if type(means) is not list else means
                intensity_fit = np.zeros(spectralvar_fit.shape, float)
                for (height, mean, width) in zip(heights, means, widths):
                    intensity_fit += gaussian(spectralvar_fit, height, mean, width)
            if show_all_species_fit:
                interpolation = interp1d(spectralvar, spectra[i][:,3],
                                         kind='quadratic')
                intensity_fit_all = interpolation(spectralvar_fit)
       
        ax_col, ax_row = None, None
        if join_subplots:
            idx_col, idx_row = None, None
            num_col_i = 1 + i % num_cols
            exceptions_inds = exceptions_indices_int + exceptions_indices_spec
            if i not in exceptions_inds:
                if num_row_i > 1:
                    idx_col = num_col_i - 1
                    ax_col = axes[idx_col]
                if num_col_i > 1:
                    idx_row = (num_row_i - 1) * num_cols
                    ax_row = axes[idx_row]
            if (i+1) % num_cols == 0:
                num_row_i += 1

        ax = plt.subplot(num_rows, num_cols, i+1, sharex=ax_col, sharey=ax_row)
        if not use_common_labels and i+1 == 1 and len(intensity_lims) == 0:
            ax.set_ylabel(intensity_label, labelpad=ypad)
        axes += [ax]
        if input_spectral_variable == 'velocity':
            spectralvar -= velocity_offset
            spectralvar_fit -= velocity_offset
        if input_spectral_variable == 'velocity' and use_frequency:
            spectralvar = rest_freqs[i] * (1 - spectralvar/speed_light)
            spectralvar_fit = rest_freqs[i] * (1 - spectralvar_fit/speed_light)
        elif input_spectral_variable == 'frequency' and not use_frequency:
            spectralvar = speed_light[i] * (1-spectralvar/rest_freqs[i])
            spectralvar_fit = speed_light[i] * (1-spectralvar_fit/rest_freqs[i]) 
            spectralvar -= velocity_offset
            spectralvar_fit -= velocity_offset
        spectralvar *= spectral_factor
        intensity *= intensity_factor
        plt.step(spectralvar, intensity, where='mid', color=[0.1]*3, lw=lw)
        if fill_spectrum:
            plt.fill_between(spectralvar, intensity, step='mid',
                             color=light_gray)
            plt.axhline(y=0, color=0.9*np.array(light_gray),
                        lw=0.5*lw, zorder=1.)
        if show_quantum_numbers or show_rest_species_lines:
            y1, y2 = plt.ylim()
            margin = (1.4 if show_quantum_numbers and show_rest_species_lines
                      else 0.8)
            y2 += margin*(y2-y1)
        if len(plot_fits) > 0 and plot_fits[i]:
            spectralvar_fit *= spectral_factor
            intensity_fit *= intensity_factor
            if fit_style == 'steps':
                plt.step(spectralvar_fit, intensity_fit, where='mid',
                         color=fit_colors[i], lw=flw)
                if show_all_species_fit:
                    plt.step(spectralvar_fit, intensity_fit_all, where='mid',
                             color=all_species_fit_color, lw=flw)
            elif fit_style in ['lines', 'curve']:
                plt.plot(spectralvar_fit, intensity_fit, color=fit_colors[i],
                         lw=flw, alpha=0.8)
                if show_all_species_fit:
                   plt.plot(spectralvar_fit, intensity_fit_all,
                            color=all_species_fit_color, lw=flw)
            show_extra_fits = extra_fits != [] or 'additional fits' in config_entry
            if show_extra_fits:
                extra_fit_params = (extra_fits if extra_fits != []
                                    else config_entry['additional fits'])
                for extra_entry in extra_fit_params:
                    show_fit = (extra_entry['show'] if 'show' in extra_entry
                                else True)
                    if show_fit:
                        color = (extra_entry['color'] if 'color' in extra_entry
                                 else fit_colors[i])
                        file_prefix = extra_entry['file']
                        file_prefix_split = file_prefix.split(separator)
                        extra_folder = separator.join(file_prefix
                                        .split(separator)[:-1]) + separator
                        file_prefix = file_prefix_split[-1]
                        extra_file = list(Path(folder + entry + data_subfolder
                           + extra_folder).glob('**/{}*'.format(file_prefix)))
                        if len(extra_file) == 0:
                            raise Exception('No files for additional fits for '
                                            'molecule {} found.'.format(entry))
                        extra_file = str(extra_file[0])
                        data = np.loadtxt(extra_file)
                        spectralvar = data[:,0]
                        spectralvar_fit = copy.copy(spectralvar)
                        intensity_fit = data[:,2]
                        if fit_style == 'curve':
                            spectralvar_fit = np.linspace(spectralvar[0],
                                         spectralvar[-1], ssf*len(spectralvar))
                            if gaussian_fit:
                                params, _, _ = multigaussian_fit(spectralvar,
                                                spectralvar_fit, verbose=False)
                                intensity_fit = multigaussian(spectralvar_fit,
                                                              *params)
                            else:
                                interpolation = interp1d(spectralvar,
                                               intensity_fit, kind='quadratic')
                                intensity_fit = interpolation(spectralvar_fit)
                        rest_freq = float(extra_file.split('_')[-1]) / 1e9
                        if input_spectral_variable == 'velocity':
                            spectralvar_fit -= velocity_offset
                        if (input_spectral_variable == 'velocity'
                                and use_frequency):
                            spectralvar_fit = \
                                rest_freq * (1-spectralvar_fit/speed_light)
                        elif (input_spectral_variable == 'frequency'
                              and not use_frequency):
                            spectralvar_fit = \
                                speed_light * (1-spectralvar_fit/rest_freqs[i])
                            spectralvar_fit -= velocity_offset
                        spectralvar_fit *= spectral_factor
                        intensity_fit *= intensity_factor
                        if fit_style == 'steps':
                            plt.step(spectralvar_fit, intensity_fit, zorder=1.,
                                     where='mid', color=color, lw=flw)
                        elif fit_style in ['lines', 'curve']:
                            plt.plot(spectralvar_fit, intensity_fit, zorder=1.,
                                     color=color, lw=flw, alpha=0.8)
            if 'legend' in config_entry:
                legend_params = config_entry['legend']
                legend_font_size = (legend_params['font size'] if 'font size'
                                    in legend_params else 0.7*label_font_size)
                for element in legend_params['elements']:
                    label = format_text_with_species_name(element['label'])
                    plt.plot([], color=element['color'], label=label)
                loc = (legend_params['location']
                       if 'location' in legend_params else 'best')
                ncols = (legend_params['number of columns']
                         if 'number of columns' in legend_params else 1)
                plt.legend(loc=loc, ncols=ncols, fontsize=legend_font_size)
        if show_transitions:
            if len(np.array(transitions_main[i]).shape) == 1:
                transitions_main[i] = [transitions_main[i]]
            if titles[i] == 'auto':
                titles[i] = format_species_name(transitions_main[i][0][1],
                                                acronyms=acronyms)
            lines += [[]]
            transitions = []
            if show_main_species_lines:
                transitions += [transitions_main[i]]
            transitions = transitions_main[i]
            if show_rest_species_lines:
                if len(np.array(transitions_rest[i]).shape) == 1:
                    transitions += [transitions_rest[i]]
                else:
                    for transition_j in transitions_rest[i]:
                        transitions += [transition_j]
            if 'transitions threshold' in config['species'][i][entry]:
                lines_lim_i = config_entry['transitions threshold']
            elif lines_lim == 'auto':
                ylims = (np.array(config_entry['intensity limits'])*intensity_factor
                          if 'intensity limits' in config_entry else plt.ylim())
                irow = i // num_rows
                lines_lim_i = 0.1 * intensity_lims[irow][1]
            else:
                lines_lim_i = float(lines_lim)
            name_main = transitions_main[i][0][1]
            for line in transitions:
                x0 = float(line[0])
                name = line[1]
                line_intensity = (float(line[5])
                                  if len(line) > 5 and line[5] != '' else 0.)
                if input_spectral_variable == 'velocity':
                    x0 -= velocity_offset
                if input_spectral_variable == 'velocity' and use_frequency:
                    x0 = rest_freqs[i] * (1 - x0 / speed_light)
                elif input_spectral_variable == 'frequency' and not use_frequency:
                    x0 = speed_light * (1 - x0 / rest_freqs[i])
                    x0 -= velocity_offset
                x0 *= spectral_factor
                label = ''
                if show_species_names:
                    molecule = format_species_name(name, acronyms=acronyms)
                    label += molecule
                    if show_quantum_numbers:
                        label += ':  '
                if show_quantum_numbers:
                    label += format_quantum_numbers(line[2])
                is_uplim = line[7] if len(line) > 7 else False
                is_uplim = True if is_uplim == 'true' else False
                if (name == name_main
                        or (not is_uplim and line_intensity > lines_lim_i)):
                    lines[i] += [{'position': x0, 'molecule': name,
                                  'label': label, 'intensity': line_intensity}]
        plt.margins(x=0)
        plt.minorticks_on()
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=2)
        if join_subplots:
            plt.tick_params(axis='y', pad=5.)
            plt.tick_params(axis='x', pad=5.)
        elif (use_frequency and len(frequency_lims) == 0
              or not use_frequency and len(spectral_lims) == 0):
            plt.tick_params(axis='y', pad=8.)
        if len(titles[i]) > 0 and show_subplot_titles:
            plt.text(0.05, subtitles_height, titles[i], transform=ax.transAxes,
                      horizontalalignment='left', verticalalignment='top',
                      fontsize=label_font_size,
                      bbox=dict(boxstyle='round', color='white', alpha=0.8,
                                linewidth=0., zorder=3.))
        
    fig.align_ylabels()
    title = format_text_with_species_name(figure_titles[f])
    plt.suptitle(title, fontweight='semibold', y=title_height)
    
    # Limits and axis.
    
    for (i, intensity_lim) in zip(y_idx, intensity_lims):
        ylims = []
        if i <= len(spectra) - num_cols:
            num_cols_i = num_cols
        else:
            num_cols_i = len(spectra) % num_cols
        if intensity_lim == 'auto':
            for j in range(num_cols_i):
                if i+j not in exceptions_indices_int:
                    ylims += [axes[(i+j)].get_ylim()]
            ylims = np.array(ylims)
            ylims = np.array([min(ylims[:,0]), max(ylims[:,1])])
            if show_transitions:
                if show_rest_species_lines:
                    ylims[1] *= 1.5
                if show_quantum_numbers:
                    ylims[1] *= 2
        else:
            ylims = intensity_lim
        ylims = np.array(ylims) * intensity_factor
        for j in range(num_cols_i):
            axes[i+j].set_ylim(ylims)
            if j != 0 and i+j not in exceptions_indices_int:
                plt.setp(axes[i+j].get_yticklabels(), visible=False)
        if not use_common_labels:
            axes[i].set_ylabel(intensity_label, labelpad=ypad)

    for (i, spectral_lim) in enumerate(spectral_lims):
        xlims = []
        for j in range(num_rows):
            if i + j*num_cols < len(spectra):
                num_rows_i = num_rows
            else:
                num_rows_i = len(spectra) % num_rows
        if spectral_lim == 'auto':
            for j in range(num_rows_i):
                if i+j*num_cols not in exceptions_indices_spec:
                    xlims += [axes[(i+j*num_cols)].get_xlim()]
            xlims = np.array(xlims)
            xlims = [min(xlims[:,0]), max(xlims[:,1])]
        else:
            xlims = spectral_lim  
        xlims = np.array(xlims) * spectral_factor
        for j in range(num_rows_i):
            idx = i+j*num_cols
            if use_frequency:
                axes[idx].xaxis.set_major_locator(ticker.MaxNLocator(3))
            else:
                axes[idx].set_xlim(xlims)
        
    if not join_subplots:
        for (i, entry) in enumerate(config['species']):
            name = list(entry.keys())[0]
            if 'frequency limits' in entry[name]:
                axes[i].set_xlim(entry[name]['frequency limits'])
            if 'velocity limits' in entry[name]:
                axes[i].set_xlim(entry[name]['velocity limits'])
            if 'intensity limits' in entry[name]:
                ylims = entry[name]['intensity limits']
                axes[i].set_ylim(ylims)            
    
    if use_common_labels:
        fig.supxlabel(spectral_label, fontsize=font_size)
        fig.supylabel(intensity_label, fontsize=font_size)
    else:
        for i in x_idx:
            axes[i].set_xlabel(spectral_label, labelpad=xpad)
        
    # Exceptions.
    for (i, ylim) in zip(exceptions_indices_int, exceptions_limits_int):
        ylim = np.array(ylim) * intensity_factor
        axes[i].set_ylim(ylim)
        axes[i].tick_params(axis='y', pad=3.)
        if join_subplots and (i % num_cols + 1) == num_cols:
            axes[i].yaxis.tick_right()
    # axes[i].yaxis.set_ticks_position('both')
    for (i, xlim) in zip(exceptions_indices_spec, exceptions_limits_spec):
        xlim = np.array(xlim) * spectral_factor
        axes[i].set_xlim(xlim)
        axes[i].tick_params(axis='x', pad=3.)
        if join_subplots and i // num_rows == 0:
            axes[i].xaxis.tick_top()
    # axes[i].xaxis.set_ticks_position('both')
    
    # Transition lines.
    if show_transitions:
        for (i, lines_i) in enumerate(lines):
            positions_i = [line_ij['position'] for line_ij in lines_i]
            inds = np.argsort(positions_i)
            lines_i = np.array(lines_i)[inds].tolist()
            entry = list(config['species'][i].keys())[0]
            if len(lines_i) == 0:
                molecule = format_species_name(entry)
                axes[i].text(0.05, subtitles_height, molecule,
                        transform=axes[i].transAxes, ha='left', va='top',
                        fontsize=label_font_size,
                        bbox=dict(boxstyle='round', color='white', alpha=0.8,
                                                     linewidth=0., zorder=3.))
                continue
            if 'label font size' in config['species'][i][entry]:
                lfs = config['species'][i][entry]['label font size']
            else:
                lfs = label_font_size
            # plt.subplot(num_rows, num_cols, i+1)
            ylim1, ylim2 = axes[i].get_ylim()
            yrange = ylim2 - ylim1
            xlims = axes[i].get_xlim()
            xlim1, xlim2 = xlims
            xrange = xlim2 - xlim1
            xlim1_pix, ylim1_pix = axes[i].transData.transform([xlim1, ylim1])
            xlim2_pix, ylim2_pix = axes[i].transData.transform([xlim2, ylim2])
            yrange_pix = ylim2_pix - ylim1_pix
            positions, labels, molecules = [], [], []
            for (j, line) in enumerate(lines_i):
                if line['label'] in labels:
                    for position in positions:
                        positions_j = [line['position'], positions[-1]]
                        overlap, _, _ = check_overlap(positions_j, xlims,
                                                      label_min_dist)
                        if overlap:
                            break
                else:
                    overlap = False
                if not overlap:
                    positions += [line['position']]
                    labels += [line['label']]
                    molecules += [line['molecule']]
            label_positions = create_label_positions(positions,
                                plot_interval=xlims, width=label_min_dist)
            overlap, overlap_mask, overlap_group_inds = \
                check_overlap(positions, xlims, label_min_dist)
            overlap_inds = np.arange(len(positions))[overlap_mask]
            line_ylims, plotted_labels = {}, []
            for j in range(len(labels)):
                x0 = positions[j]
                label = labels[j]
                eps = 0.0
                plot_cond = x0 > xlim1 + eps*xrange and x0 < xlim2 - eps*xrange
                if plot_cond:
                    x0_label = label_positions[j]
                    if (not show_rest_species_lines
                            and x0_label < xlims[0] + xrange/4):
                        y0_label = 0.82 * ylim2
                    else:
                        y0_label = 0.95 * ylim2
                    current_species = transitions_main[i][0][1]
                    color = ('black' if current_species in molecules[j]
                             else other_species_color)
                    plot_transition = (show_main_species_lines
                              if color == 'black' else show_rest_species_lines)
                    if plot_transition:
                        if not mark_lines:
                            y0_label_or = copy.copy(y0_label)
                            text = axes[i].text(x=x0_label, y=y0_label, s=label,
                                    rotation=90, fontsize=lfs, alpha=0.,
                                    ha='left', va='top', color=color, zorder=2.)
                            x1_pix, x2_pix, y1_pix, y2_pix = \
                                parse_text_location(text, fig)
                            spectralvar = spectra[i][:,0]
                            intensity = spectra[i][:,1]
                            inds = np.argsort(spectralvar)
                            y0_label = np.interp(x0_label,
                                            spectralvar[inds], intensity[inds])
                            y0_label += 1.3 * ((y2_pix - y1_pix) / yrange_pix
                                               * yrange)
                            y0_label = min(y0_label_or, y0_label)
                        text = axes[i].text(x=x0_label, y=y0_label, s=label,
                                        rotation=90, fontsize=lfs, zorder=2.,
                                        ha='center', va='top', color=color)
                        x1_pix, x2_pix, y1_pix, y2_pix = \
                            parse_text_location(text, fig)
                        ymin = ylim1 / yrange
                        if not (show_species_names or show_quantum_numbers):
                            ymax = 1.0
                        else:
                            ymax = (y0_label/yrange - 0.03
                                    - (y2_pix - y1_pix) / yrange_pix)
                        if mark_lines and j not in overlap_inds:
                            x = np.array([x0, x0])
                            y = np.array([ymin, ymax]) * yrange
                            axes[i].plot(x, y, color=gray, lw=0.9*lw, ls='--',
                                         alpha=0.7)
                        line_ylims[j] = [ymin, ymax]
            for inds in overlap_group_inds:
                group_line_positions = []
                group_label_positions = []
                group_limits = []
                for k in inds:
                    x0 = positions[k]
                    if plot_cond and k in line_ylims.keys():
                        group_line_positions += [positions[k]]
                        group_label_positions += [label_positions[k]]
                        group_limits += [line_ylims[k]]
                group_limits = np.array(group_limits)
                if mark_lines and plot_transition and len(group_limits) > 0:
                    x0 = np.mean(group_line_positions)
                    ymin = np.min(group_limits[:,0])
                    ymax = np.min(group_limits[:,1])
                    ynode = ymin + 0.85*(ymax - ymin)
                    x = np.array([x0, x0])
                    y = np.array([ymin, ynode]) * yrange
                    axes[i].plot(x, y, color=gray, lw=0.9*lw, ls='--',
                                 alpha=0.7)
                    for label_position, limit in zip(group_label_positions,
                                                     group_limits):
                        xl = label_position
                        ymax = limit[1]
                        x = np.array([x0, xl])
                        y = np.array([ynode, ymax]) * yrange
                        axes[i].plot(x, y, color=gray, lw=0.9*lw, ls='--',
                                     alpha=0.7)
            _, ymax = axes[i].get_ylim()
            axes[i].margins(y=0)
            axes[i].set_ylim(top=ymax)
    
    # Exporting of the figure.
    if save_figure:
        image_name = (figure_titles[f].replace('-','').replace('  ',' ').
                      replace(' ','-').replace('*',''))
        if len(image_name) == 0:
            image_name = 'lines'
        image_name += '.pdf'
        plt.savefig(image_name, bbox_inches='tight')
        print('Saved figure in {}'.format(image_name))

print()
plt.show()

plt.rcParams.update(plt.rcParamsDefault)