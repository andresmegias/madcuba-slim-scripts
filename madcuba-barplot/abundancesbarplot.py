#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Abundances Bar Plotter
----------------------
Version 1.2

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

config_file = 'barplot4r.yaml'

import os
import re
import sys
import copy
import itertools
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import richvalues as rv

plt.rcParams.update({'font.size': 12})

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
            if y1[i].center > y2[i].center:
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
        plt.bar(xi, y1i.center, color=c1, edgecolor='black', width=width,
                align='center', lw=1, hatch=hatch1)
        plt.errorbar(xi-xm, y1i.center, np.array(y1i.unc).reshape(2,-1),
                     uplims=y1i.is_uplim, fmt='.', capsize=2, capthick=1,
                     ms=3, lw=1, color='k')
        if len(y2) > 0:
            if y2i.is_uplim:
                hatch2 = '/'
                y2i.set_lims_factor(2)
            else:
                hatch2 = ''
            plt.bar(xi, y2i.center, color=c2, edgecolor='black', width=width,
                    align='center', lw=1, hatch=hatch2)
            plt.errorbar(xi+xm, y2i.center, np.array(y2i.unc).reshape(2,-1),
                         uplims=y2i.is_uplim, fmt='.', capsize=2, capthick=1,
                         ms=3, lw=1, color='k')

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


def cosine_similarity(u, v):
    """
    Compute the cosine similarity between the input detection vectors.

    Parameters
    ----------
    u,v : list / array
        Input vectors, each for one different source. For each molecule and
        position, the value is 1 if the species is detected and 0 otherwise.

    Returns
    -------
    d : float
        Resulting value, ranging from 0 to 1. If 1, the input vectors are
        identical. If 0, they are completely different.
    """
    if len(u) != len(v):
        raise Exception('Input vectors should have the same length.')
    d = sum([1. if u_i == v_i else 0. for u_i,v_i in zip(u,v)])
    d /= u.size
    return d

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
    'compute cosine similarity': False,
    'compute mean abundance': False,
    'compute mean molecular mass': False,
    'use upper limits for calculations': True
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
compute_cosine_similarity = config['compute cosine distance']
compute_mean_mol_mass = config['compute mean molecular mass']
compute_mean_abund = config['compute mean abundance']
consider_uplims = config['use upper limits for calculations']

# Font sizes.
legend_font_size = (0.9*font_size if type(legend_font_size) is None
                   else legend_font_size)
plt.rcParams.update({'font.size': font_size})

#%%

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
    y_a += [rv.rich_array(data_a['abundance'], domain=[0,np.inf], num_sf=1)]
    labels_a += [data_a['molecule'].values]
    source = file_a.replace('.csv','').split('-')[0]
    sources += [source]
    abunds[source] = {}
    abunds[source][name_a[-1]] = rv.rich_array(y_a[-1], num_sf=1)
    if len(entry) == 1:
        y_b += [[]]
        color_b += ['tab:gray']
        name_b += [' ']
    elif len(entry) == 2:
        file_b = list(entry.keys())[1]
        color_b += [entry[file_b]['color']]
        name_b += [entry[file_b]['name']]
        data_b = pd.read_csv(file_b, comment='#')
        y_b += [rv.rich_array(data_b['abundance'].values, domain=[0,np.inf],
                              num_sf=1)]
        labels_b += [data_b['molecule'].values]
        abunds[source][name_b[-1]] = rv.rich_array(y_b[-1], num_sf=1)
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
        bibarplot(x+off+i*width, y_a[i], y_b[i],
                  color_a[i], color_b[i], width=width,
                  point_spacing=point_spacing)
        alpha_b = 1
    else:
        bibarplot(x+off+i*width, y_a[i], '', color_a[i], '',
                  width=width, point_spacing=0)
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
plt.xticks(ticks=x+0.52, labels=labels, rotation=90)
plt.tick_params(axis='x', which='both', bottom=False)
plt.ylabel('abundance relative to H$_2$')
plt.tight_layout()

# Export image.
plt.savefig(output_file)
print('\nSaved figure in {}.'.format(output_file))

# Cosine similarity.
if set(name_b) != {' '}:
    positions = list(set(name_a) & set(name_b))
else:
    positions = list(set(name_a))
if ' ' in name_a and ' ' in name_b:
    compute_cosine_similarity = False
cosine_similarity_params = list(itertools.combinations(sources, 2))
if compute_cosine_similarity:
    vectors = {}
    for source in abunds:
        vectors[source] = []
        for position in positions:
            for abund in abunds[source][position]:
                value = 1 if not abund.is_uplim else 0
                vectors[source] += [value]
        vectors[source] = np.array(vectors[source])
    cosine_similarities = {}
    for param in cosine_similarity_params:
        name1, name2 = param
        cosine_similarities['{}-{}'.format(name1, name2)] = \
            rv.RichValue(cosine_similarity(vectors[name1], vectors[name2]),
                         unc=1/len(vectors[name1]), num_sf=2)
    print('\nCosine similarity.')
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
            cond = (np.ones(len(molecules), bool) if consider_uplims
                    else ~ abunds[source][position].are_uplims)
            cond *= [np.isfinite(abund.center)
                     for abund in abunds[source][position]]
            # for i, abund in enumerate(abunds[source][position]):
            #     if abund.is_uplim:
            #         abunds[source][position][i].is_uplim = False
            #         abunds[source][position][i].is_range = False
            #         abunds[source][position][i].unc = [0, 0]
            #         # abunds[source][position][i].center = 1e-13
            mean_abund_i = \
                rv.rich_fmean(abunds[source][position][cond],
                              function=np.log, inverse_function=np.exp,
                              weights=molecular_masses[cond],
                              domain=[0,np.inf], num_sf=1)
            mean_abunds[source][position] = mean_abund_i
    print('\nMean abundance.')
    for source in mean_abunds:
        mean_abund_source = []
        for position in mean_abunds[source]:
            mean_abund_i = mean_abunds[source][position]
            mean_abund_i.check_interval()
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
            cond = (np.ones(len(molecules), bool) if consider_uplims
                    else ~ abunds[source][position].are_uplims)
            cond *= [np.isfinite(abund.center)
                     for abund in abunds[source][position]]
            def log_weights(weights):
                abunds_vals = abunds[source][position][cond].centers
                log_vals = np.log10(abunds_vals)
                min_log, max_log = min(log_vals), max(log_vals)
                y = np.log10(weights) - min_log + (1/4) * (max_log - min_log)
                y = np.maximum(0., y)
                return y
            mean_mol_mass_i = \
                rv.rich_fmean(molecular_masses[cond],
                              weights=abunds[source][position][cond],
                              weight_function=log_weights,
                              domain=[0,np.inf], num_sf=1)
            mean_mol_masses[source][position] = mean_mol_mass_i
    print('\nMean molecular mass. (g/mol)')
    for source in mean_mol_masses:
        mean_mol_mass_source = []
        for position in mean_mol_masses[source]:
            mean_mol_mass_i = mean_mol_masses[source][position]
            mean_mol_mass_i.check_interval()
            print('{} - {}: {}'.format(source, position, mean_mol_mass_i))
            mean_mol_mass_source += [mean_mol_mass_i]
        if len(list(mean_mol_masses[source].keys())) > 1:
            print('{} - (mean): {}'.format(source, np.mean(mean_mol_mass_source)))

print()
plt.show()