#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MADCUBA Lines Plotter
---------------------
Version 1.3

Copyright (C) 2022 - Andrés Megías Toledano

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

config_file = 'examples/L1517B.yaml'

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

def fit_gaussians(x, y, windows, width):
    """
    Fit gaussian profiles to the data in the specified windows.

    Parameters
    ----------
    x : array
        Independent variable.
    y : array
        Dependent variable.
    windows : array
        Windows coordinates.
    margin : float
        Number of extra points of the surroundings of the windows used.

    Returns
    -------
    fit_params : list
        Parameters of the fitted gaussians.
    """
    fit_params = []
    dx = np.median(np.diff(x))
    for x1, x2 in windows:
        cond = cond = (x > x1 - width*dx) * (x < x2 + width*dx)
        params, _, _ = multigaussian_fit(x[cond], y[cond])
        fit_params += [params]
    return fit_params

def format_species_name(input_name, simplify_numbers=True):
    """
    Format text as a molecule name, with subscripts and upperscripts.

    Parameters
    ----------
    input_name : str
        Input text.
    simplify_numbers : bool, optional
        Remove the brackets between single numbers.

    Returns
    -------
    output_name : str
        Formatted molecule name.
    """
    original_name = copy.copy(input_name)
    # upperscripts
    possible_upperscript, in_upperscript = False, False
    output_name = ''
    upperscript = ''
    inds = []
    for i, char in enumerate(original_name):
        if (char.isupper() and not possible_upperscript
                and '-' in original_name[i:]):
            inds += [i]
            possible_upperscript = True
        elif char.isupper():
            inds += [i]
        if char == '-' and not in_upperscript:
            inds += [i]
            in_upperscript = True
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
    output_name = output_name.replace('^$_', '$^').replace('$$', '')
    if output_name.endswith('+') or output_name.endswith('-'):
        symbol = output_name[-1]
        output_name = output_name.replace(symbol, '$^{'+symbol+'}$')
    original_name = copy.copy(output_name)
    # subscripts
    output_name, subscript, prev_char = '', '', ''
    in_bracket = False
    for i,char in enumerate(original_name):
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
    # vibrational numbers
    output_name = output_name.replace(',vt', ', vt')
    output_name = output_name.replace(', vt=', '$, v_t=$')
    output_name = output_name.replace(',v=', '$, v=$')
    for i,char in enumerate(output_name):
        if output_name[i:].startswith('$v_t=$'):
            output_name = output_name[:i+5] + \
                output_name[i+5:].replace('$_','').replace('_$','')
    # some formatting
    output_name = output_name.replace('$$', '').replace('__', '')
    # remove brackets from single numbers
    if simplify_numbers:
        single_numbers = re.findall('{(.?)}', output_name)
        for number in set(single_numbers):
            output_name = output_name.replace('{'+number+'}', number)
    return output_name

def format_transition_numbers(input_numbers):
    """
    Format text as a molecular transition.

    Parameters
    ----------
    input_numbers : str
        Input text with the transition numbers separated with commas (,) and
        hyphens (-).

    Returns
    -------
    output_numbers : str
        Formatted transition numbers.
    """
    if '-' in input_numbers and not ',' in input_numbers:
        numbers = input_numbers.split('-')
        num_qn = len(numbers) // 2
        input_numbers = ','.join(numbers[:num_qn]) + '-' + \
            ','.join(numbers[num_qn:])
    if '-' in input_numbers and ',' in input_numbers:
        number1, number2 = input_numbers.split('-')
        first_number = number1.split(',')[0]
        subindices = ','.join(number1.split(',')[1:])
        number1 = '$' + first_number + '_{' + subindices + '}$'
        first_number = number2.split(',')[0]
        subindices = ','.join(number2.split(',')[1:])
        number2 = '$' + first_number + '_{' + subindices + '}$'
        output_numbers = '-'.join([number1, number2])
        output_numbers = output_numbers.replace('$-$', ' \\rightarrow ')
    else:
        output_numbers = input_numbers.replace('-', '$\\rightarrow$')
    return output_numbers

def check_overlap(positions, plot_interval, width=1/20):
    """
    Check if any of the given label positions overlap.
    
    Parameters
    ----------
    positions : list (float)
        Positions to check.
    plot_interval : float
        Horizontal range of the plot in which the labels will appear.
    width : float (optional)
        Minimum separation betweeen positions to check the overlaping.
        The default is 1/20.

    Returns
    -------
    overlap : bool.
        If True, there is overlaping of at least two positions.
    overlap_mask : list (int)
        Indices of the positions which overlap.
    """
    positions = np.array(positions)
    num_positions = len(positions)
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

def create_label_positions(initial_positions, plot_interval, width=1/20,
                           max_iters=3000):
    """
    Generates label positions that not overlap from the given positions.

    Parameters
    ----------
    initial_positions : list (float)
        Initial positions.
    plot_interval : float
        Horizontal range of the plot in which the labels will appear.
    width : float (optional)
        Minimum separation betweeen labels to check the overlaping.
        The default is 1/20.
    max_iters : int (optional)
        Maximum number of iterations for finding the proper positions.

    Returns
    -------
    new positions : list (float).
        Generated positions, that not overlap.
    """
    initial_order = np.argsort(initial_positions)
    plot_range = plot_interval[1] - plot_interval[0]
    overlap, overlap_mask, _, = check_overlap(initial_positions, plot_interval)
    if not overlap:
        return initial_positions
    else:
        for i in range(max_iters):
            factor = 0.3*i/1000
            new_positions = []
            for j,x in enumerate(initial_positions):
                if overlap_mask[j]:
                    xx = np.random.normal(x, factor*width*plot_range)
                    xx = max(plot_interval[0], xx)
                    xx = min(xx, plot_interval[1])
                else:
                    xx = x
                new_positions += [xx]
            overlap, overlap_mask, _, = \
                check_overlap(new_positions, plot_interval)
            new_order = np.argsort(new_positions)
            if not np.array_equal(initial_order, new_order):
                overlap = True
            if not overlap:
                break
        if not overlap:
            return new_positions
        else:
            return initial_positions   

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

#%%

# Default options.
default_options = {
    'figure size': 'auto',
    'font size': 10,
    'label font size': None,
    'frame width': 0.8,
    'plot line width': 1.2,
    'fit line width': None,
    'rows': 1,
    'columns': 1,
    'subplot titles height': 0.90,
    'ticks direction': 'in',
    'use frequencies': False,
    'velocity limits': [],
    'frequency limits': [],
    'intensity limits': [],
    'velocity label': 'velocity (km/s)',
    'frequency label': 'frequency (GHz)',
    'intensity label': 'intensity (K)',
    'fit plot style': 'steps',
    'gaussian fit': False,
    'fit color': 'tab:red',
    'all species fit color': 'tab:blue',
    'other species color': 'tab:blue',
    'figure titles': [],
    'data folders': [],
    'species': [],
    'join subplots': True,
    'show transitions': False,
    'show transition numbers': False,
    'show all species transitions': False,
    'save figure': True,
    'show all species fit': False,
    'transition labels minimum distance': 0.05
    }
gray = tuple([0.6]*3)
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
label_font_size = (0.8*font_size if type(label_font_size) is None
                   else label_font_size)
frame_width = config['frame width']
line_width = config['plot line width']
fit_line_width = config['fit line width']
fit_line_width = line_width if type(fit_line_width) is None else fit_line_width
num_rows = config['rows']
num_cols = config['columns']
subtitles_height = config['subplot titles height']
join_subplots = config['join subplots']
ticks_direction = config['ticks direction']
use_frequency = config['use frequencies']
velocity_lims = config['velocity limits']
if len(velocity_lims) != num_cols:
    velocity_lims = ['auto'] * num_cols
frequency_lims = config['frequency limits']
if len(frequency_lims) != num_cols:
    frequency_lims = ['auto'] * num_cols
intensity_lims = config['intensity limits']
if len(intensity_lims) != num_rows:
    intensity_lims = ['auto'] * num_rows
velocity_label = config['velocity label']
frequency_label = config['frequency label']
intensity_label = config['intensity label']
fit_style = config['fit plot style']
gaussian_fit = config['gaussian fit']
usual_fit_color = config['fit color']
all_species_fit_color = config['all species fit color']
other_species_color = config['other species color']
show_transitions = config['show transitions']
show_all_species_transitions = config['show all species transitions']
show_transition_numbers = config['show transition numbers']
save_figure = config['save figure']
label_min_dist = config['transition labels minimum distance']
show_all_species_fit = config['show all species fit']
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
if (('frequency limits' in config_or or 'frequency label' in config_or)
        and ('velocity limits' not in config_or
             and 'velocity label' not in config_or)):
    use_frequency = True
if use_frequency:
    join_subplots = False
speed_light = 2.99e5  # km/s

# Graphical options.
lw = line_width
flw = fit_line_width
plt.rcParams.update({'font.size': font_size})
plt.rcParams.update({'axes.linewidth': frame_width})
for i in ['x','y']:
    plt.rcParams.update({i+'tick.major.width': frame_width})
    plt.rcParams.update({i+'tick.minor.width': 0.8*frame_width})
plt.rcParams['xtick.direction'] = ticks_direction
plt.rcParams['ytick.direction'] = ticks_direction
plt.rcParams['xtick.major.size'] = 5.
plt.rcParams['ytick.major.size'] = 5.
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['axes.formatter.useoffset'] = False

#%%

print()
print('MADCUBA Lines Plotter')
print('---------------------')
print()
    
for f,folder in enumerate(folders):
    
    if not folder.endswith(separator):
        folder += separator
        
    # Data files.
    spectrum_files, titles = [], []
    plot_fits, fit_colors, manual_fits, manual_lines = [], [], [], []
    exceptions_indices_vel, exceptions_indices_int = [], []
    exceptions_limits_vel, exceptions_limits_int = [], []
    for i, molecule in enumerate(config['species']):
        molecule = list(molecule.keys())[0]
        all_spectra = Path(folder + molecule + data_subfolder).glob('**/*')
        all_spectra = sorted([str(pp) for pp in all_spectra])
        if len(all_spectra) == 0:
            raise Exception('No files for molecule {}.'.format(molecule))
        for spectrum in all_spectra:
            spectrum = spectrum.split('/')[-1]
            config_prefix = config['species'][i][molecule]['file']
            if type(config_prefix) == list:
                config_prefix = config_prefix[f]
            if spectrum.startswith(config_prefix):
                spectrum_files += [folder + molecule + data_subfolder + spectrum]
                if 'title' in config['species'][i][molecule]:
                    title = config['species'][i][molecule]['title']
                    title = title.replace('\n ','\n')
                    if title.count('*') == 2:
                        species_name = title.split('*')[1]
                        title = title.replace('*{}*'.format(species_name),
                                              format_species_name(species_name))
                else:
                    title = 'auto'
                titles += [title]
                if 'fit' in config['species'][i][molecule]:
                    config_fit = config['species'][i][molecule]['fit']
                else:
                    config_fit = False
                if type(config_fit) == list:
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
                    if type(manual_lines_i) is list:
                        manual_lines_i = manual_lines_i[f]
                    elif (type(manual_lines_i) is list
                          and len(manual_lines_i) == 1):
                        manual_lines_i = manual_lines_i[0]
                else:
                    manual_lines_i = {}
                manual_lines += [manual_lines_i]
                
        if 'intensity limits' in config['species'][i][molecule]:
            if (i+1 - num_cols) % num_cols == 0:
                exceptions_indices_int += [i]
                exceptions_limits_int += \
                    [config['species'][i][molecule]['intensity limits']]
            else:
                raise Exception ('Wrong place of exception in molecule {}.'
                                 .format(molecule))
        if 'velocity limits' in config['species'][i][molecule]:
            if i < num_cols:
                exceptions_indices_vel += [i]
                exceptions_limits_vel += \
                    [config['species'][i][molecule]['velocity limits']]
            else:
                raise Exception ('Wrong place of exception in molecule {}.'
                                 .format(molecule))
        if 'fit color' in config['species'][i][molecule]:
            config_color = config['species'][i][molecule]['fit color']
            if type(config_color) is list:
                config_color = config_color[f]
            fit_colors += [config_color]
        else:
            fit_colors += [usual_fit_color]
    
    # Loading of the spectra.
    spectra, spectra_all, transitions_main, transitions_rest = [], [], [], []
    freqs = []
    for file in spectrum_files:
        spectra += [np.loadtxt(file)]
        freqs += [float(file.split('_')[-1]) / 1e9]
        if show_all_species_fit:
            file = file.replace('/' + molecule, '/' + molecule + '_all')
            spectra_all += [np.loadtxt(file)]
        file = file.replace(data_subfolder, spectroscopy_subfolder)
        transitions_main += [np.loadtxt(file, str)]
        prefix = file.split(spectroscopy_subfolder)[0] + spectroscopy_subfolder
        texts = file.split(spectroscopy_subfolder)[1].split('_')
        if show_all_species_transitions:
            file = '_'.join([texts[0], 'TRANSITIONS_ALL', *texts[2:]])
            file = prefix + file
            transitions_rest += [np.loadtxt(file, str)]
    spectra = spectra[:num_cols*num_rows]
    num_rows = len(spectra) // num_cols + int(len(spectra) % num_cols != 0)
    intensity_lims = intensity_lims[:num_rows]
    for i, transition_group in enumerate(transitions_main):
        transitions_main[i] = list(transition_group)
        if len(transition_group.shape) == 1:
            transition = transition_group
            transitions_main[i] = list(transition)
            if len(transition) == 3:
                transitions_main[i].insert(2, '')
        else:
            for j, transition in enumerate(transition_group):
                transitions_main[i][j] = list(transition)
                if len(transition) == 3:
                    transitions_main[i][j].insert(2, '')
    for i, transition_group in enumerate(transitions_rest):
        transitions_rest[i] = list(transition_group)
        if len(transition_group.shape) == 1:
            transition = transition_group
            transitions_rest[i] = list(transition)
            if len(transition) == 3:
                transitions_rest[i].insert(2, '')
        else:
            for j, transition in enumerate(transition_group):
                transitions_rest[i][j] = list(transition)
                if len(transition) == 3:
                    transitions_rest[i][j].insert(2, '') 
    
    #%% Plot.
    
    # Figure.
    if figure_size == 'auto':
        figure_size = (3*num_cols, 2*num_rows)
    fig = plt.figure(1+f, figsize=figure_size)
    plt.clf()
    # Indices of rows and columns.
    y_idx = np.arange(0, len(spectra), num_cols)
    x_idx = np.arange(len(spectra))
    x_idx = x_idx[x_idx+1 > len(spectra) - num_cols]
    
    # Plot of the data.
    axes = []
    lines = []
    for i, spectrum in enumerate(spectra):
        velocity = spectrum[:,0]
        intensity = spectrum[:,1]
        intensity_fit = spectrum[:,2]
        if show_all_species_fit:
            intensity_fit_all = spectra_all[i][:,2]
        velocity_fit = velocity.copy()
        if fit_style == 'curve':
            N = spectrum.shape[0]
            velocity_fit = np.linspace(velocity[0], velocity[-1], ssf*N)
            if len(manual_fits) > 0 and not manual_fits[i]:
                if gaussian_fit:
                    # guess = [intensity_fit.max(), velocity[N//2], 0.3]
                    # params, _ = curve_fit(gaussian, velocity, intensity_fit,
                    #                       p0=guess)
                    params, _, _ = multigaussian_fit(velocity, intensity_fit,
                                                      verbose=False)
                    intensity_fit = multigaussian(velocity_fit, *params)
                else:
                    interpolation = interp1d(velocity, intensity_fit,
                                             kind='quadratic')
                    intensity_fit = interpolation(velocity_fit)
            else:
                widths = np.array(manual_lines[i]['width'], float) / 2.355
                heights = np.array(manual_lines[i]['intensity'], float)
                if 'position' in manual_lines[i]:
                    means = np.array(manual_lines[i]['position'], float)
                else:
                    means = float(transitions_main[i][0])
                widths = [widths] if type(widths) is not list else widths
                heights = [heights] if type(heights) is not list else heights
                means = [means] if type(means) is not list else means
                intensity_fit = np.zeros(velocity_fit.shape, float)
                for height, mean, width in zip(heights, means, widths):
                    intensity_fit += gaussian(velocity_fit,
                                              height, mean, width)
            if show_all_species_fit:
                interpolation = interp1d(velocity, spectra_all[i][:,2],
                                         kind='quadratic')
                intensity_fit_all = interpolation(velocity_fit)
        ax = plt.subplot(num_rows, num_cols, i+1)
        if i+1 == 1 and len(intensity_lims) == 0:
            plt.ylabel(intensity_label)
        axes += [ax]
        if join_subplots:
            plt.subplots_adjust(wspace=0, hspace=0)
        if use_frequency:
            frequency = freqs[i] * (1 + velocity / speed_light)
            frequency_fit = freqs[i] * (1 + velocity_fit / speed_light)
            velocity = frequency
            velocity_fit = frequency_fit
            velocity_label = frequency_label
            velocity_lims = frequency_lims
        plt.step(velocity, intensity, where='mid', color='black', lw=lw)
        if len(plot_fits) > 0 and plot_fits[i]:
            if fit_style == 'steps':
                plt.step(velocity_fit, intensity_fit, where='mid',
                         color=fit_colors[i], lw=flw)
                if show_all_species_fit:
                    plt.step(velocity_fit, intensity_fit_all, where='mid',
                             color=all_species_fit_color, lw=flw)
            elif fit_style in ['lines', 'curve']:
                plt.plot(velocity_fit, intensity_fit, color=fit_colors[i],
                         lw=flw, alpha=0.8)
                if show_all_species_fit:
                   plt.plot(velocity_fit, intensity_fit_all,
                            color=all_species_fit_color, lw=flw)
        if len(np.array(transitions_main[i]).shape) == 1:
            transitions_main[i] = [transitions_main[i]]
        if titles[i] == 'auto':
            titles[i] = format_species_name(transitions_main[i][0][1])
        lines += [[]]
        transitions = transitions_main[i]
        if show_all_species_transitions:
            if len(np.array(transitions_rest[i]).shape) == 1:
                transitions += [transitions_rest[i]]
            else:
                for transition_j in transitions_rest[i]:
                    transitions += [transition_j]
        for line in transitions:
            x0 = float(line[0])
            if use_frequency:
                x0 = freqs[i] * (1 + x0 / speed_light)
            molecule = format_species_name(line[1])
            label = '' 
            if show_transition_numbers:
                label += format_transition_numbers(line[2])
            if show_all_species_transitions:
                prefix = molecule
                if show_transition_numbers:
                    prefix += ':  '
                label = prefix + label
            lines[i] += [{'position': x0, 'label': label}]
        plt.margins(x=0)
        plt.minorticks_on()
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=2)
        if join_subplots:
            plt.tick_params(axis='y', pad=5.)
            plt.tick_params(axis='x', pad=5.)
        elif (use_frequency and len(frequency_lims) == 0
              or not use_frequency and len(velocity_lims) == 0):
            plt.tick_params(axis='y', pad=8.)
        if len(titles[i]) > 0 and not show_all_species_transitions:
            plt.text(0.05, subtitles_height, titles[i], transform=ax.transAxes,
                      horizontalalignment='left', verticalalignment='top')

    fig.align_ylabels()
    y_st = 0.93 if join_subplots else 0.98
    plt.suptitle(figure_titles[f], fontweight='semibold', y=y_st)
    
    # Limits and axis.
    
    for i, intensity_lim in zip(y_idx, intensity_lims):
        ylims = []
        if i <= len(spectra) - num_cols:
            num_cols_i = num_cols
        else:
            num_cols_i = len(spectra)%num_cols
        if intensity_lim == 'auto':
            for j in range(num_cols_i):
                if i+j not in exceptions_indices_int:
                    ylims += [axes[(i+j)].get_ylim()]
            ylims = np.array(ylims)
            ylims = [min(ylims[:,0]), max(ylims[:,1])]
            if show_transitions:
                if show_all_species_transitions:
                    ylims[1] *= 1.5
                if show_transition_numbers:
                    ylims[1] *= 2
        else:
            ylims = intensity_lim  
        for j in range(num_cols_i):
            if join_subplots:
                axes[i+j].set_ylim(ylims)
                if j != 0 and i+j not in exceptions_indices_int:
                    axes[i+j].set_yticklabels([])
        axes[i].set_ylabel(intensity_label, labelpad=7.)

    for i, velocity_lim in zip(range(num_cols), velocity_lims):
        xlims = []
        for j in range(num_rows):
            if i + j*num_cols < len(spectra):
                num_rows_i = num_rows
            else:
                num_rows_i = len(spectra)%num_rows
        if velocity_lim == 'auto':
            for j in range(num_rows_i):
                if i+j*num_cols not in exceptions_indices_vel:
                    xlims += [axes[(i+j*num_cols)].get_xlim()]
            xlims = np.array(xlims)
            xlims = [min(xlims[:,0]), max(xlims[:,1])]
        else:
            xlims = velocity_lim  
        for j in range(num_rows_i):
            idx = i+j*num_cols
            if use_frequency:
                axes[idx].xaxis.set_major_locator(ticker.MaxNLocator(3))
            else:
                axes[idx].set_xlim(xlims)
           
    for i in x_idx:
        axes[i].set_xlabel(velocity_label, labelpad=7.)
        
    # Exceptions.
    for i, ylim in zip(exceptions_indices_int, exceptions_limits_int):
        axes[i].set_ylim(ylim)
        axes[i].yaxis.tick_right()
        axes[i].tick_params(axis='y', pad=3.)
    # axes[i].yaxis.set_ticks_position('both')
    for i, xlim in zip(exceptions_indices_vel, exceptions_limits_vel):
        axes[i].set_xlim(xlim)
        axes[i].xaxis.tick_top()
        axes[i].tick_params(axis='x', pad=3.)
    # axes[i].xaxis.set_ticks_position('both')
    
    # Transition lines.
    if show_transitions:
        for i,lines_i in enumerate(lines):
            # plt.subplot(num_rows, num_cols, i+1)
            ylim1, ylim2 = axes[i].get_ylim()
            yrange = ylim2 - ylim1
            xlims = axes[i].get_xlim()
            xlim1, xlim2 = xlims
            xrange = xlim2 - xlim1
            xlim1_pix, ylim1_pix = axes[i].transData.transform([xlim1, ylim1])
            xlim2_pix, ylim2_pix = axes[i].transData.transform([xlim2, ylim2])
            yrange_pix = ylim2_pix - ylim1_pix
            xrange_pix = xlim2_pix - xlim1_pix
            positions, labels = [], []
            for line in lines[i]:
                positions += [line['position']]
                labels += [line['label']]
            label_positions = \
                create_label_positions(positions, plot_interval=xlims,
                                       width=label_min_dist)
            overlap, overlap_mask, overlap_groups_inds = \
                check_overlap(positions, plot_interval=xlims,
                              width=label_min_dist)
            overlap_inds = np.arange(len(positions))[overlap_mask]
            line_ylims = {}
            for j,line in enumerate(lines_i):
                x0 = positions[j]
                if x0 > xlim1 + 0.01*xrange and x0 < xlim2 - 0.01*xrange:
                    x0_label = label_positions[j]
                    label = line['label']
                    if (not show_all_species_transitions
                        and x0_label < xlims[0] + xrange/4):
                        y0_label = 0.82 * ylim2
                    else:
                        y0_label = 0.95 * ylim2
                    if titles[i] in label:
                        color = 'black'
                    else:
                        color = other_species_color
                    text = axes[i].text(x=x0_label, y=y0_label, s=label,
                                        rotation=90, fontsize=label_font_size,
                                        alpha=1., ha='center', va='top',
                                        color=color)
                    x1_pix, x2_pix, y1_pix, y2_pix = \
                        parse_text_location(text, fig)
                    ymin = ylim1 / yrange
                    if (not show_all_species_transitions
                        and not show_transition_numbers):
                        ymax = 1.0
                    else:
                        ymax = (y0_label/yrange - 0.03
                                - (y2_pix - y1_pix) / yrange_pix)
                    if j not in overlap_inds:
                        x = np.array([x0, x0])
                        y = np.array([ymin, ymax]) * yrange
                        axes[i].plot(x, y, color=gray, lw=0.9*lw, ls='--',
                                     alpha=0.7)
                    line_ylims[j] = [ymin, ymax]
            for group in overlap_groups_inds:
                group_line_positions = []
                group_label_positions = []
                group_limits = []
                for k in group:
                    x0 = positions[k]
                    if x0 > xlim1 + 0.01*xrange and x0 < xlim2 - 0.01*xrange:
                        group_line_positions += [positions[k]]
                        group_label_positions += [label_positions[k]]
                        group_limits += [line_ylims[k]]
                group_limits = np.array(group_limits)
                if len(group_limits) > 0:
                    line_position = np.mean(group_line_positions)
                    ymin = min(group_limits[:,0])
                    ymax = min(group_limits[:,1])
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
            
    if not join_subplots:
        plt.tight_layout()
    
    # Exporting of the figure.
    if save_figure:
        image_name = (figure_titles[f].replace('-','').replace('  ',' ').
                      replace(' ','-'))
        if len(image_name) == 0:
            image_name = 'lines'
        image_name += '.pdf'
        plt.savefig(image_name, bbox_inches='tight')
        print('Saved figure in {}'.format(image_name))

print()
plt.show()

# Restore the default graphical options.
plt.rcParams['font.size'] = plt.rcParamsDefault['font.size']
plt.rcParams['axes.linewidth'] = plt.rcParamsDefault['axes.linewidth']
for i in ['x','y']:
    plt.rcParams[i+'tick.major.width'] = \
        plt.rcParamsDefault[i+'tick.major.width']
    plt.rcParams[i+'tick.minor.width'] = \
        plt.rcParamsDefault[i+'tick.minor.width']
plt.rcParams['xtick.direction'] = plt.rcParamsDefault['xtick.direction']
plt.rcParams['ytick.direction'] = plt.rcParamsDefault['ytick.direction']
plt.rcParams['xtick.major.size'] = plt.rcParamsDefault['xtick.major.size'] 
plt.rcParams['ytick.major.size'] = plt.rcParamsDefault['ytick.major.size']
plt.rcParams['ytick.right'] = plt.rcParamsDefault['ytick.right']
plt.rcParams['xtick.top'] = plt.rcParamsDefault['xtick.top']
