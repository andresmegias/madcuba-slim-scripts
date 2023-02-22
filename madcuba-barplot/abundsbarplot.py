#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abundances Bar Plotter
----------------------
Version 1.6

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

config_file = 'examples/barplot3.yaml'

import os
import re
import sys
import copy
import math
import itertools
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import richvalues as rv

def bibarplot(x, y1, y2, color1, color2, width, point_spacing=0.1):
    """
    Create a bar plot with double-height bars.

    Parameters
    ----------
    x : array
        List of positions of the bars.
    y1 : rich carray
        Array containing the height of the first bars as rich values.
    y2 : rich array
        Array containing the height of the first bars as rich values.
    color1 : str
        Color of the first bars.
    color2 : str
        Color of the second bars.
    width : float
        Width of the bars.
    point_spacing : float, optional
        Spacing between bars. The default is 0.1.

    Returns
    -------
    None.
    """
    xm = point_spacing
    for i in range(len(y1)):
        xi = x[i] + width/2
        if len(y2) > 0:
            if y1[i].main > y2[i].main:
                y1i, y2i = y1[i], y2[i]
                c1, c2 = color1, color2
            else:
                y1i, y2i = y2[i], y1[i]
                c1, c2 = color2, color1
        else:
            y1i = y1[i]
            c1 = color1
        if y1i.is_uplim:
            hatch1 = '/'
            y1i.set_lims_factor(2)
        else:
            hatch1 = ''
        plt.bar(xi, y1i.main, color=c1, edgecolor='black', width=width,
                align='center', lw=1, hatch=hatch1)
        plt.errorbar(xi-xm, y1i.main, y1i.unc_eb(), uplims=y1i.is_uplim,
                     fmt='.', capsize=2, capthick=1, ms=3, lw=1, color='k')
        if len(y2) > 0:
            if y2i.is_uplim:
                hatch2 = '/'
                y2i.set_lims_factor(2)
            else:
                hatch2 = ''
            plt.bar(xi, y2i.main, color=c2, edgecolor='black', width=width,
                    align='center', lw=1, hatch=hatch2)
            plt.errorbar(xi+xm, y2i.main, y2i.unc_eb(), uplims=y2i.is_uplim,
                         fmt='.', capsize=2, capthick=1, ms=3, lw=1, color='k')

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
    output_name = output_name.replace('^$_', '$^').replace('$$', '')
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

def count_atoms(molecule, atoms=['H', 'D', 'C', 'O', 'N', 'S', 'P']):
    """
    Count the number of atoms in the given molecule.

    Parameters
    ----------
    molecule : str
        Chemical formula of the molecule.
    atoms : list, optional
        List containing the atoms to be counted (their names should be just one
        letter). The default is ['H', 'D', 'C', 'N', 'O', 'P', 'S'].

    Returns
    -------
    num_atoms : dict
        Dictionary containing the number of each atom.
    """
    atoms = np.array(atoms)
    num_atoms = {atom: 0 for atom in atoms}
    prev_char = ''
    for i,char in enumerate(molecule):
        if char in atoms:
            num_atoms[char] += 1
        if char.isdigit() and prev_char in atoms:
            num_atoms[prev_char] += int(char) - 1
        prev_char = char
    return num_atoms
        
def molecular_mass(molecule):
    """
    Compute the molecular mass of the given molecule.

    Parameters
    ----------
    molecule : str
        Chemical formula of the molecule.

    Returns
    -------
    molec_mass : float
        Molecular mass of the molecule.
    """
    molecular_masses = {'H': 1., 'D': 2., 'C': 12., 'N': 14., 'O': 16.,
                        'P': 31., 'S': 32.}
    num_atoms = count_atoms(molecule)
    molec_mass = 0
    for atom in molecular_masses:
        molec_mass += num_atoms[atom] * molecular_masses[atom]
    return molec_mass

def fractional_similarity(u, v):
    """
    Compute the fractional similarity between the input detection vectors.

    Parameters
    ----------
    u,v : list / array
        Input vectors, each for a different source. For each molecule and
        position, the value is 1 if the species is detected and 0 otherwise.

    Returns
    -------
    fs : rich value
        Resulting value, ranging from 0 to 1. If 1, the input vectors are
        identical. If 0, they are completely different.
    """
    if len(u) != len(v):
        raise Exception('Input vectors should have the same length.')
    s = (np.array(u) == np.array(v)).sum() / len(u)
    s_ = np.array([])
    for i in [0,1]:
        for j in range(len(u)):
            uv = (copy.copy(u), copy.copy(v))
            uv[i][j] = int(not bool(uv[i][j]))
            uij, vij = uv[0], uv[1]
            sij = (np.array(uij) == np.array(vij)).sum() / len(u)
            s_ = np.append(s_, sij)
    uncs = [np.mean(s - s_[s_<s]), np.mean(s_[s_>s] - s)]
    s = rv.RichValue(s, uncs, domain=[0,1])
    s.num_sf = 2
    return s

def cosine_similarity(u, v):
    """
    Compute the cosine similarity between the input arrays.

    Parameters
    ----------
    u,v : list / array
        Input vectors, each for a different source, containing the abundances
        of different molecules.

    Returns
    -------
    s : rich value
        Resulting value, ranging from 0 to 1. If 1, the input vectors are
        identical. If 0, they are completely different.
    """
    if len(u) != len(v):
        raise Exception('Input vectors should have the same length.')
    s = np.dot(u,v) / (np.dot(u,u)**0.5 * np.dot(v,v)**0.5)
    s = min(1,s)
    s_ = np.array([])
    for j in [0,1]:
        for k in range(len(u)):
            uv = (copy.copy(u), copy.copy(v))
            uv[j][k] = int(not bool(uv[j][k]))
            ui, vi = uv[0], uv[1]
            si = np.dot(ui,vi) / (np.dot(ui,ui)**0.5 * np.dot(vi,vi)**0.5)
            s_ = np.append(s_, si)
    s_1 = s - s_[s_<s]
    s_2 = s_[s_>s] - s
    s_1 = [0] if len(s_1) == 0 else s_1
    s_2 = [0] if len(s_2) == 0 else s_2
    uncs = [np.mean(s_1), np.mean(s_2)]
    s = rv.RichValue(s, uncs, domain=[0,1])
    s.num_sf = 2
    return s

def angular_similarity(u, v):
    """
    Compute the angular similarity between the input arrays.

    Parameters
    ----------
    u,v : list / array
        Input vectors, each for a different source, containing the abundances
        of different molecules.

    Returns
    -------
    s : rich value
        Resulting value, ranging from 0 to 1. If 1, the input vectors are
        identical. If 0, they are completely different.
    """
    if len(u) != len(v):
        raise Exception('Input vectors should have the same length.')
    s = np.dot(u,v) / (np.dot(u,u)**0.5 * np.dot(v,v)**0.5)
    s = min(1,s)
    s = 1 - np.arccos(s) / (math.tau/4)
    s_ = np.array([])
    for j in [0,1]:
        for k in range(len(u)):
            uv = (copy.copy(u), copy.copy(v))
            uv[j][k] = int(not bool(uv[j][k]))
            ui, vi = uv[0], uv[1]
            si = np.dot(ui,vi) / (np.dot(ui,ui)**0.5 * np.dot(vi,vi)**0.5)
            si = 1 - np.arccos(si) / (math.tau/4)
            s_ = np.append(s_, si)
    s_1 = s - s_[s_<s]
    s_2 = s_[s_>s] - s
    s_1 = [0] if len(s_1) == 0 else s_1
    s_2 = [0] if len(s_2) == 0 else s_2
    uncs = [np.mean(s_1), np.mean(s_2)]
    s = rv.RichValue(s, uncs, domain=[0,1])
    s.num_sf = 2
    return s

def rich_cosine_similarity(u, v):
    """
    Compute the cosine similarity between the input rich arrays.

    Parameters
    ----------
    u,v : list / array
        Input vectors, each for a different source, containing the abundances
        of different molecules.

    Returns
    -------
    s : rich value
        Resulting value, ranging from 0 to 1. If 1, the input vectors are
        identical. If 0, they are completely different.
    """
    if len(u) != len(v):
        raise Exception('Input vectors should have the same length.')
    def function(*args):
        L = len(args) // 2
        u, v = args[:L], args[L:]
        y, u2, v2 = 0, 0, 0
        for ui,vi in zip(u,v):
            y += ui*vi
            u2 += ui**2
            v2 += vi**2
        y /= (u2 * v2)**0.5
        return y
    s = rv.function_with_rich_values(function, [*u,*v], domain=[0,1])
    s.num_sf = 2
    ub = (~u.are_uplims()).astype(int)
    vb = (~v.are_uplims()).astype(int)
    ref_uncs = cosine_similarity(ub, vb).unc
    s.unc[0] = max(ref_uncs[0], s.unc[0])
    s.unc[1] = max(ref_uncs[1], s.unc[1])
    return s

def rich_angular_similarity(u, v):
    """
    Compute the angular similarity between the input rich arrays.

    Parameters
    ----------
    u,v : list / array
        Input vectors, each for a different source, containing the abundances
        of different molecules.

    Returns
    -------
    s : rich value
        Resulting value, ranging from 0 to 1. If 1, the input vectors are
        identical. If 0, they are completely different.
    """
    if len(u) != len(v):
        raise Exception('Input vectors should have the same length.')
    def function(*args):
        L = len(args) // 2
        u, v = args[:L], args[L:]
        y, u2, v2 = 0, 0, 0
        for ui,vi in zip(u,v):
            y += ui*vi
            u2 += ui**2
            v2 += vi**2
        y /= (u2 * v2)**0.5
        y = 1 - 4/math.tau * np.arccos(y)
        return y
    s = rv.function_with_rich_values(function, [*u,*v], domain=[0,1])
    s.num_sf = 2
    ub = (~u.are_uplims()).astype(int)
    vb = (~v.are_uplims()).astype(int)
    ref_uncs = angular_similarity(ub, vb).unc
    s.unc[0] = max(ref_uncs[0], s.unc[0])
    s.unc[1] = max(ref_uncs[1], s.unc[1])
    return s


#%%

# Default options.
default_options = {
    'input files': [],
    'output file': 'barplot.pdf',
    'molecules': [],
    'figure size': [12, 8],
    'font size': 12,
    'bar spacing': 0.25,
    'point spacing': 0.05,
    'legend position': [],
    'legend font size': None,
    'inferior limit': None,
    'similarity type': 'fractional',
    'compute similarity': False,
    'compute mean abundance': False,
    'compute mean molecular mass': False,
    'use upper limits for means': True
    }

# Configuration file.
if len(sys.argv) == 1:
    config_path = os.path.realpath('./' + config_file)
else:
    config_path = os.path.realpath(sys.argv[1])
config_folder = '/'.join(config_path.split('/')[:-1]) + '/'
os.chdir(config_folder)
if os.path.isfile(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
else:
    raise FileNotFoundError('Configuration file not found.')
config = {**default_options, **config}

# Options.
input_files = config['input files']
output_file = config['output file']
bar_spacing = config['bar spacing']
point_spacing = config['point spacing']
figure_size = tuple(config['figure size'])
font_size = config['font size']
legend_position = tuple(config['legend position'])
legend_font_size = config['legend font size']
inferior_limit = float(config['inferior limit'])
similarity_type = config['similarity type']
compute_similarity = config['compute similarity']
compute_mean_mol_mass = config['compute mean molecular mass']
compute_mean_abund = config['compute mean abundance']
use_uplims = config['use upper limits for calculations']

if similarity_type not in ('fractional', 'cosine', 'angular',
                           'discretized cosine', 'discretized angular'):
    raise Exception("Wrong similarity type. It can be 'fractional', 'cosine', "
                    + "'angular', 'discretized cosine' or 'discretized angular'.")

# Font sizes.
legend_font_size = (0.9*font_size if legend_font_size is None
                    else legend_font_size)
plt.rcParams.update({'font.size': font_size})

#%%

plt.close('all')
print()
print('Abundances Bar Plotter')
print('----------------------')

# Define variables.
y_a, y_b = [], []
dy_a, dy_b = [], []
color_a, color_b = [], []
name_a, name_b = [], []
labels_a, labels_b = [], []
sources = []
abunds = {}

# Get data.
for entry in input_files:
    file_a = list(entry.keys())[0]
    color_a += [entry[file_a]['color']]
    name_a += [entry[file_a]['name']]
    data_a = pd.read_csv(file_a, comment='#')
    rarr = rv.rich_array(data_a['abundance'], domain=[0,np.inf])
    rarr.num_sf = 1
    y_a += [rarr]
    labels_a += [data_a['molecule'].values]
    source = file_a.replace('.csv','').split('-')[0]
    sources += [source]
    abunds[source] = {}
    abunds[source][name_a[-1]] = rarr
    if len(entry) == 1:
        y_b += [[]]
        color_b += ['tab:gray']
        name_b += [' ']
    elif len(entry) == 2:
        file_b = list(entry.keys())[1]
        color_b += [entry[file_b]['color']]
        name_b += [entry[file_b]['name']]
        data_b = pd.read_csv(file_b, comment='#')
        rarr = rv.rich_array(data_b['abundance'].values, domain=[0,np.inf])
        rarr.num_sf = 1
        y_b += [rarr]
        labels_b += [data_b['molecule'].values]
        abunds[source][name_b[-1]] = rarr
    else:
        raise Exception('Error: The number of sources should be 1 or 2.')
       
all_labels = labels_a + labels_b
labels_nonsorted = all_labels[0]
for i in range(len(all_labels)-1):
    labels_nonsorted = list(set(labels_nonsorted) & set(all_labels[i+1]))

labels_lens = [len(labels_i) for labels_i in all_labels]
labels_ref = all_labels[np.argmax(labels_lens)].astype(str)

cond = [i for i,label in enumerate(labels_ref) if label in labels_nonsorted]
labels = labels_ref[cond]

for j,labels_a_i in enumerate(labels_a):
    cond = [i for i,label in enumerate(labels_a_i) if label in labels_nonsorted]
    y_a[j] = y_a[j][cond]
    abunds[sources[j]][name_a[j]] = abunds[sources[j]][name_a[j]][cond]
for j,labels_b_i in enumerate(labels_b):
    cond = [i for i,label in enumerate(labels_b_i) if label in labels_nonsorted]
    y_b[j] = y_b[j][cond]
    abunds[sources[j]][name_b[j]] = abunds[sources[j]][name_b[j]][cond]

# Format species names.
labels = labels.astype(object)
for i,label in enumerate(labels):
    labels[i] = format_species_name(label)
N = len(labels)          
x = np.arange(N)

# Figure.
plt.figure(1, figsize=figure_size)
plt.clf()
# Bar plot options.
n = len(input_files)
m = bar_spacing
width = (1-m)/n
off = m / 2
point_spacing *= width/2
# Bar plots.
for i,entry in enumerate(input_files):
    if len(entry) == 2:
        bibarplot(x+off+i*width, y_a[i], y_b[i], color_a[i], color_b[i],
                  width=width, point_spacing=point_spacing)
        alpha_b = 1
    else:
        bibarplot(x+off+i*width, y_a[i], '', color_a[i], '', width=width,
                  point_spacing=0)
        alpha_b = 0
    title = list(entry.keys())[0].split('-')[0]
    plt.plot([], [], alpha=0, label=title+':')
    plt.plot([], [], color=color_a[i], label=name_a[i])
    plt.plot([], [], color=color_b[i], label=name_b[i], alpha=alpha_b)
# Plot options.
plt.legend(ncol=n, loc='upper center', bbox_to_anchor=legend_position,
           framealpha=0, fontsize=legend_font_size)
plt.yscale('log')
plt.xlim([-off, N+off])
plt.ylim(bottom=inferior_limit)
plt.gca().yaxis.set_ticks_position('both')
plt.xticks(ticks=x+0.52, labels=labels, rotation=90)
plt.tick_params(axis='x', which='both', bottom=False)
plt.ylabel('abundance relative to H$_2$')
plt.tight_layout()

# Export image.
plt.savefig(output_file)
print('\nSaved figure in {}.'.format(output_file))

# Similarity.
if similarity_type == 'fractional':
    similarity_function = fractional_similarity
elif similarity_type == 'discretized cosine':
    similarity_function = cosine_similarity
elif similarity_type == 'discretized angular':
    similarity_function = angular_similarity
elif similarity_type == 'cosine':
    similarity_function = rich_cosine_similarity
elif similarity_type == 'angular':
    similarity_function = rich_angular_similarity
if set(name_b) != {' '}:
    names = np.array([name_a, name_b])
    positions = set(names[:,0])
    for i in range(len(name_a)-1):
        positions &= set(names[:,i+1])
    common_positions = list(positions)
else:
    common_positions = list(set(name_a))
if ' ' in name_a and ' ' in name_b:
    compute_similarity = False
similarity_params = list(itertools.combinations(sources, 2))
if compute_similarity:
    vectors = {}
    for source in abunds:
        vectors[source] = []
        if similarity_type in ('fractional',
                               'discretized cosine', 'discretized angular'):
            positions = list(abunds[source].keys())
            if len(positions) == 2 and len(common_positions) == 2:
                for position in positions:
                    for abund in abunds[source][position]:
                        value = 1 if not abund.is_uplim else 0
                        value = np.nan if np.isnan(abund.main) else value
                        vectors[source] += [value]
            else:
                for abund1, abund2 in zip(abunds[source][positions[0]],
                                          abunds[source][positions[1]]):
                    value = (1 if not (abund1.is_uplim and abund2.is_uplim)
                             else 0)
                    value = (np.nan if all(np.isnan([abund1.main, abund2.main]))
                             else value)
                    vectors[source] += [value]
        else:
            for position in common_positions:
                vectors[source] += [*abunds[source][position].function(np.log)]
        vectors[source] = (np.array(vectors[source]) if similarity_type in
                    ('fractional', 'discretized cosine', 'discretized angular')
                    else rv.rich_array(vectors[source]))
    cosine_similarities = {}
    use_uplims_for_similarity = \
        (True if similarity_type in ('fractional', 'discretized cosine',
                                     'discretized angular') else use_uplims)
    for param in similarity_params:
        name1, name2 = param
        num_entries = len(vectors[name1])
        if num_entries != 0:
            cond = (np.ones(num_entries, bool) if use_uplims_for_similarity
                    else ~vectors[name1].are_uplims())
            cond *= [np.isfinite(rv.rich_value(entry).main)
                     for entry in vectors[name1]]
            cond = (cond * np.ones(num_entries, bool)
                    if use_uplims_for_similarity
                    else cond * ~vectors[name2].are_uplims())
            cond *= [np.isfinite(rv.rich_value(entry).main)
                     for entry in vectors[name2]]
            cos_sim = similarity_function(vectors[name1][cond],
                                          vectors[name2][cond])
        else:
            cos_sim = None
        cosine_similarities['{}-{}'.format(name1, name2)] = cos_sim
    print('\n{} similarity.'.format(similarity_type.capitalize()))
    for group in cosine_similarities:
        print('- {}: {}'.format(group, cosine_similarities[group]))

# Mean abundance.
if compute_mean_abund:
    molecules = [label.replace('$_','').replace('$^','').replace('$','')
                 for label in labels]
    molecular_masses = np.array([molecular_mass(molecule)
                                 for molecule in molecules])
    mean_abunds = {}
    for source in abunds:
        mean_abunds[source] = {}
        for position in abunds[source]:
            cond = (np.ones(len(molecules), bool) if use_uplims
                    else ~ abunds[source][position].are_uplims())
            cond *= [np.isfinite(abund.main)
                     for abund in abunds[source][position]]
            mean_abund_i = rv.rich_fmean(abunds[source][position][cond],
                             function=np.log, inverse_function=np.exp,
                             weights=molecular_masses[cond], domain=[0,np.inf],
                             consider_ranges=False)
            mean_abunds[source][position] = mean_abund_i
    print('\nMean abundance.')
    for source in mean_abunds:
        mean_abund_source = []
        for position in mean_abunds[source]:
            mean_abund_i = mean_abunds[source][position]
            print('{} - {}: {}'.format(source, position, mean_abund_i))
            mean_abund_source += [mean_abund_i]
        if len(list(mean_abunds[source].keys())) > 1:
            print('{} - (mean): {}'.format(source, np.mean(mean_abund_source)))

# Mean molecular mass.
if compute_mean_mol_mass:
    molecules = [label.replace('$_','').replace('$^','').replace('$','')
                 for label in labels]
    molecular_masses = np.array([molecular_mass(molecule)
                                 for molecule in molecules])
    mean_mol_masses = {}
    for source in abunds:
        mean_mol_masses[source] = {}
        for position in abunds[source]:
            weights = abunds[source][position]
            cond = (np.ones(len(molecules), bool) if use_uplims
                    else ~ abunds[source][position].are_uplims())
            cond *= [np.isfinite(abund.main)
                     for abund in abunds[source][position]]
            def log_weights(weights):
                abunds_vals = abunds[source][position][cond].mains()
                log_vals = np.log10(abunds_vals)
                min_log, max_log = min(log_vals), max(log_vals)
                y = np.log10(weights) - min_log + (1/4) * (max_log - min_log)
                y = np.maximum(0., y)
                return y
            mean_mol_mass_i = rv.rich_fmean(molecular_masses[cond],
                                weights=abunds[source][position][cond],
                                weight_function=log_weights, domain=[0,np.inf])
            mean_mol_masses[source][position] = mean_mol_mass_i
    print('\nMean molecular mass. (g/mol)')
    for source in mean_mol_masses:
        mean_mol_mass_source = []
        for position in mean_mol_masses[source]:
            mean_mol_mass_i = mean_mol_masses[source][position]
            print('{} - {}: {}'.format(source, position, mean_mol_mass_i))
            mean_mol_mass_source += [mean_mol_mass_i]
        if len(list(mean_mol_masses[source].keys())) > 1:
            print('{} - (mean): {}'.format(source,
                                           np.mean(mean_mol_mass_source)))

print()
plt.show()

plt.rcParams['font.size'] = plt.rcParamsDefault['font.size']