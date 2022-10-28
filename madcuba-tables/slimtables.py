#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MADCUBA Table Generator
-----------------------
Version 1.7

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

config_file = 'example/L1517B.yaml'

import os
import re
import sys
import copy
import yaml
import platform
import numpy as np
import pandas as pd
import richvalues as rv

def combine_rms(v):
    """
    Return the final rms of the sum of signals with the given rms noises.

    Parameters
    ----------
    v : list / array (float)
        Individual rms noises of each signal.

    Returns
    -------
    y : float
        Resulting rms noise.
    """
    y = np.sqrt(1 / np.sum(1/np.array(v)**2))
    return y

def merge_tables(*args):
    """
    Combine the input dataframes into one.

    Parameters
    ----------
    args : dataframes
        Input dataframes to be joined.

    Returns
    -------
    final_table : dataframe
        Resulting dataframe.
    """
    final_table = {}
    for table, columns in args:
        for column in columns:
            name = column
            if name in final_table:
                name += '_'
            final_table[name] = table[column].to_numpy()
    final_table = pd.DataFrame(final_table)
    return final_table

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
    if '-' in input_numbers:
        prev_char = ''
        for i,char in enumerate(input_numbers):
            if i>0 and char == '-' and prev_char != ',':
                numbers1 = input_numbers[:i]
                numbers2 = input_numbers[i+1:]
                break
            prev_char = char
        first_number = numbers1.split(',')[0]
        subindices = ','.join(numbers1.split(',')[1:])
        numbers1 = '$' + first_number + '_{' + subindices + '}$'
        first_number = numbers2.split(',')[0]
        subindices = ','.join(numbers2.split(',')[1:])
        numbers2 = '$' + first_number + '_{' + subindices + '}$'
        output_numbers = '-'.join([numbers1, numbers2])
        output_numbers = output_numbers.replace('$-$', ' \\rightarrow ')
    else:
        output_numbers = input_numbers
    return output_numbers

#%%       
     
light_speed = 2.998E5 # km/s

default_config = {
    'create abundances table': False,
    'create lines table': False,
    'export abundances list': False,
    'input files': {
        'MADCUBA table': '',
        'processing table': '',
        'LaTeX template': '',
        'non-LTE lines table': {}
        },
    'output files': {
        'abundances list': '',
        'LaTeX file': 'tables.tex'
        },
    'molecules': [],
    'non-LTE molecules': [],
    'reference column density (/cm2)': 1,
    'lines (MHz)': 'auto',
    'tables scientific notation': {},
    'lines margin (MHz)': 0.2,
    'frequency decimals': 3,
    'S/N threshold': 3,
    'multiplying factors': {},
    'recalculate area uncertainty': True,
    'correct false detections': True
    }
default_sections = {
    'multiplying factors': {
    'abundances table': {}, 'lines table': {}}
    }
default_tables_template = \
r"""
\documentclass[10pt,spanish]{article}
\usepackage{mathpazo}
\usepackage{courier}
\usepackage[LGR,T1]{fontenc}
\usepackage[latin9]{inputenc}
\usepackage[a4paper]{geometry}
\geometry{verbose,tmargin=1.5cm,bmargin=1.5cm,lmargin=1.5cm,rmargin=1.5cm,headheight=0cm,headsep=0cm,footskip=0.7cm,columnsep=0.8cm}
\setlength{\parskip}{\medskipamount}
\setlength{\parindent}{0pt}
\usepackage{babel}
\addto\shorthandsspanish{\spanishdeactivate{~<>}}
\usepackage{amssymb}
\usepackage{setspace}
\setstretch{1.2}
\usepackage[unicode=true,
 bookmarks=true,bookmarksnumbered=false,bookmarksopen=false,
 breaklinks=false,pdfborder={0 0 1},backref=false,colorlinks=false]
 {hyperref}
\makeatletter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LyX specific LaTeX commands.
%% Because html converters don't know tabularnewline
\providecommand{\tabularnewline}{\\}
\@ifundefined{date}{}{\date{}}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% User specified LaTeX commands.
\usepackage{babel}
\usepackage{pdflscape}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\fnum@figure}{Figure~\thefigure}
\renewcommand{\fnum@table}{Table~\thetable}
\usepackage[margin=10pt,font=footnotesize,labelfont=bf,labelsep=period,justification=justified]{caption}
%justification=centering
\usepackage[skip=4pt]{caption}
\decimalpoint
\usepackage{titlesec}
%\titlespacing*{\section}{0pt}{3.5ex plus 1ex pt minus .2ex}{2.3ex plus .2ex}
%\titlespacing*{\subsection}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}
%\titlespacing*{\subsection}{0pt}{3.25ex plus 1ex minus .2ex}{1.5ex plus .2ex}
\titlespacing*{\section}{0pt}{2.5ex plus .2ex minus .2ex}{0.5ex plus 0ex}
\titlespacing*{\subsection}{0pt}{2.5ex plus .2ex minus .2ex}{0.2ex plus 0ex}
\titlespacing*{\subsubsection}{0pt}{2.5ex plus .2ex minus .2ex}{0.1ex plus 0ex}
\setlength{\floatsep}{0.7\baselineskip plus 0pt minus 0pt}
\setlength{\textfloatsep}{0.7\baselineskip plus 0pt minus 0pt}
\setlength{\intextsep}{0.7\baselineskip plus 0pt minus 0pt}
\addtolength{\skip\footins}{5pt plus 0pt}
\let\@fnsymbol\@arabic
\@addtoreset{footnote}{page}
\renewcommand{\thefootnote}{\ifcase\value{footnote}\or*\or**\or***\or
\#\or\#\#\or\#\#\#\fi}
\makeatother
\begin{document}
\begin{landscape}\par 
\begin{table}[h]
\caption{Transition lines table.}
\begin{tabular}{ccccccccccc}
\hline 
--tablelines_vars-- \tabularnewline
\hline 
--tablelines--  \tabularnewline
\hline 
\end{tabular}
\end{table}
\par \par \par \par 
\begin{table}[h]
\caption{Abundances table.}
\begin{tabular}{ccccccc}
\hline 
--tableabunds_vars-- \tabularnewline
\hline 
--tableabunds--  \tabularnewline
\hline 
\end{tabular}
\end{table}
\par \par \par \par \par
\end{landscape} 
\end{document}    
"""

#%%

print()
print('SLIM Table Generator')
print('--------------------')

# Folder separator.
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
        config = yaml.safe_load(file)
else:
    raise FileNotFoundError('Configuration file not found.')
    
config = {**default_config, **config}

for section in ['multiplying factors']:
    for table in ['abundances table', 'lines table']:
        if table not in config[section]:
            config[section][table] = {}
        config[section][table] = \
            {**default_sections[section][table],
             **config[section][table]}
            
export_abundances_list = config['export abundances list']
create_abundances_table = config['create abundances table']
create_lines_table = config['create lines table']
input_madcuba = config['input files']['MADCUBA table']
input_processing = config['input files']['processing table']
output_list = config['output files']['abundances list']
reference_dens = config['reference column density (/cm2)']
molecule_list = config['molecules']
nonlte_molecule_list = config['non-LTE molecules']
lines_list = config['lines (MHz)']
lines_margin = config['lines margin (MHz)']
frequency_decimals = config['frequency decimals']
sn_threshold = config['S/N threshold']
factors_abunds = config['multiplying factors']['abundances table']
factors_lines = config['multiplying factors']['lines table']
scientific_notation = config['tables scientific notation']
recalculate_area_unc = config['recalculate area uncertainty']
correct_false_detections = config['correct false detections']
if (len(scientific_notation) !=0 and 'use crosses' in scientific_notation):
    use_crosses = scientific_notation['use crosses']
else:
     use_crosses = False
if use_crosses:
    rv.defaultparams['multiplication symbol for'
                     + 'scientific notation in LaTeX'] = '\\times'

if type(input_madcuba) == list:
    num_sources = len(input_madcuba)
else:
    num_sources = 1
 

#%% 

false_lines = []
sigmas_false_lines = []

abunds_tables, lines_tables = [], []
    
if create_lines_table:
    
    lines_table_params = config['lines table']
    
    for s in range(num_sources):
        
        if type(input_madcuba) == list:
            input_madcuba_s = input_madcuba[s]
        else:
            input_madcuba_s = input_madcuba
            
        table_madcuba = pd.read_csv(input_madcuba_s)

        source = input_madcuba_s.split('-')[0]
        print('\nSource {}: {}'.format(s+1, source))

        saving_rms = False
        for element in lines_table_params:
            if 'intensity' in element[1]:
                saving_rms = True

        if recalculate_area_unc or saving_rms:
            using_table_proc = True
        else:
            using_table_proc = False

        if using_table_proc:
            if type(input_processing) == list:
                input_processing_s = input_processing[s]
            else:
                input_processing_s = input_processing
            table_proc = pd.read_csv(input_processing_s)
        
        false_lines += [{}]
        sigmas_false_lines += [{}]
        
        lines_table = pd.DataFrame({})
        
        if lines_list == 'auto':
            lines_list = {}
            previous_name = ''
            for i,row in table_madcuba.iterrows():
                if row['CHECK']:
                    name = row['Formula']
                    if name != previous_name:
                        lines_list[name] = {name: []}
                    lines_list[name][name] += [row['Frequency']]
                    previous_name = copy.copy(name)
        
        for molecule in lines_list:
            for name in lines_list[molecule]:
                transitions = table_madcuba[table_madcuba['Formula']==name]
                for line in lines_list[molecule][name]:
                    area, delta_area = 0, 0
                    intensity, delta_intensity = 0, 0
                    upper_limit = True
                    difference = abs(transitions['Frequency'] - line)
                    cond = difference < lines_margin
                    rows = transitions[cond]
                    if len(rows) == 0:
                        print('Nothing found for line {}-{}.'
                              .format(name, line))
                        transition = '...'
                        frequency = '{:.{}f}'.format(line, frequency_decimals)
                        area = np.nan
                        delta_area = np.nan
                        intensity = np.nan
                        delta_intensity = np.nan
                        width = np.nan
                        delta_width = np.nan
                        velocity = np.nan
                        delta_velocity = np.nan
                        sn_area = np.nan
                    for i, row in rows.iterrows():
                        transition = row.Transition
                        frequency = '{:.{}f}'.format(row.Frequency,
                                                     frequency_decimals)
                        upper_limit &= \
                            bool((np.sign(- row['delta N/EM']) + 1) / 2)
                        if using_table_proc:
                            line = float(row.Frequency)
                            cond = ((line > table_proc['min. frequency (MHz)'])
                                    & (line < table_proc['max. frequency (MHz)']))
                            spectra = table_proc[cond]
                        if not upper_limit:
                            width = row['Width']
                            delta_width = row['delta Width']
                            velocity = row['Velocity']
                            delta_velocity = row['delta Velocity']
                            intensity = 1e3 * row['Intensity']
                            delta_intensity = 1e3 * row['delta Intensity']
                            area += 1e3 * row['Area']
                            if using_table_proc:
                                rms = combine_rms(spectra['rms noise (mK)'])
                                resolution = \
                                    max(light_speed*spectra['resolution (MHz)']/line)
                            if recalculate_area_unc:
                                delta_area = rms * np.sqrt(resolution * width)
                            else:
                                delta_area = 1e3 * row['delta Area']
                            delta_area = float(delta_area)
                            if np.isnan(area) or np.isinf(area):
                                area = np.nan
                                delta_area = np.nan
                            else:
                                area, delta_area = \
                                    rv.round_sf_unc(area, delta_area, 1)
                            area = float(area)
                            delta_area = float(delta_area)
                            sn_area = area / delta_area
                            sn_area = float(sn_area)
                        else:
                            sn_area = np.nan
                            break
                    if sn_area <= sn_threshold:
                        upper_limit = True
                        print(('Line {}-{} is not properly detected'
                               + ' ({:.2f}-sigma).')
                              .format(name, line, float(sn_area)))
                        real_detection = True
                    else:
                        real_detection = False
                    if not name in false_lines[s]:
                        false_lines[s][name] = real_detection
                        sigmas_false_lines[s][name] = float(sn_area)
                    else:
                        false_lines[s][name] *= real_detection
                        sigmas_false_lines[s][name] = \
                            max(float(sn_area), false_lines[s][name])
                    if sn_area < 10:
                        sn_area = '{:.1f}'.format(sn_area)
                    elif sn_area >= 10:
                        sn_area = '{:.0f}'.format(sn_area)
                    for i, row in rows.iterrows():
                        if upper_limit:
                            if using_table_proc:
                                rms = combine_rms(spectra['rms noise (mK)'])
                                resolution = \
                                    max(light_speed*spectra['resolution (MHz)']
                                                 /line)
                            width = copy.copy(row['Width'])
                            if recalculate_area_unc:
                                new_area = sn_threshold * rms * \
                                    np.sqrt(resolution * width)
                            else:
                                new_area = 1e3 * row['Area']
                            new_intensity = 1e3 * row['Intensity']
                            factor = sn_threshold - sigmas_false_lines[s][name]
                            if (name in sigmas_false_lines[s]
                                    and correct_false_detections
                                    and not np.isnan(factor)):
                                new_area += delta_area * factor
                                new_intensity += delta_intensity * factor
                            area = max(area, new_area)
                            delta_area = -10
                            intensity = max(intensity, new_intensity)
                            delta_intensity = -10
                            width = np.nan
                            delta_width = np.nan
                            velocity = np.nan
                            delta_velocity = np.nan
                            sn_area = np.nan
                    intensity = rv.RichValue(intensity, delta_intensity,
                                             is_uplim=delta_intensity<0)
                    area = rv.RichValue(area, delta_area,
                                        is_uplim=delta_area<0)
                    width = rv.RichValue(width, delta_width,
                                         is_uplim=delta_width<0)
                    velocity = rv.RichValue(velocity, delta_velocity,
                                            is_uplim=delta_velocity<0)
                    params = {'species': molecule,
                              'transition': transition,
                              'frequency': frequency,
                              'intensity': intensity,
                              'area': area,
                              'width': width,
                              'velocity': velocity,
                              'sn_area': sn_area}
                    params = pd.DataFrame(params, index=[0])
                    lines_table = pd.concat([lines_table, params])
        
        for column in factors_lines:
            lines_table[column] /= float(factors_abunds[column])
            lines_table[column+'_unc'] /= float(factors_abunds[column])
        
        lines_tables += [lines_table]
        
        false_lines_species = np.array(list(false_lines[s].keys()))
        false_lines_values = np.array(list(false_lines[s].values()))
        false_detections = false_lines_species[false_lines_values == 1]
        for name in false_detections:
            print('Warning: Molecule {} is not actually detected.'.format(name))

print()

if export_abundances_list or create_abundances_table:
    
    abunds_table_params = config['abundances table']
    
    saving_abundance = False
    for element in abunds_table_params:
        if 'abundance' in element[1]:
            saving_abundance = True

    for s in range(num_sources):
    
        if type(input_madcuba) == list:
            input_madcuba_s = input_madcuba[s]
        else:
            input_madcuba_s = input_madcuba    
    
        table_madcuba = pd.read_csv(input_madcuba_s)    
    
        if saving_abundance:
            if type(output_list) == list:
                output_list_s = output_list[s]
            else:
                output_list_s = output_list
            if type(reference_dens) == list:
                reference_dens_s = rv.rich_value(reference_dens[s], num_sf=1)
            else:
                reference_dens_s = rv.rich_value(reference_dens, num_sf=1)
        else:
            reference_dens_s = rv.RichValue(1.)
        reference_dens_s.domain = [0., np.inf]
    
        N = len(molecule_list)
        inds = np.arange(1, N+1)
        abunds = []
        abunds_species = []
        
        abunds_table = {}
        table_species = []
        table_dens = []
        table_abunds = []
        table_temp = []
        
        for molecule in molecule_list:
            
            if len(molecule) == 1:

                name = list(molecule.values())[0]
                cond = table_madcuba.Formula==name
                dens = 10**table_madcuba[cond]['N/EM'].values[0]
                delta = table_madcuba[cond]['delta N/EM'].values[0]
                if len(false_lines) > 0:
                    false_detection = (name in false_lines[s]
                                       and false_lines[s][name] == 1)
                else:
                    false_detection = False
                if delta != -10 and not false_detection:
                    dens_unc = [10**delta]*2
                else:
                    dens_unc = [-10]*2
                if false_detection and correct_false_detections:
                    dens += \
                        10**delta * (sn_threshold - sigmas_false_lines[s][name])
                temp = table_madcuba[cond]['Tex/Te'].values[0]
                temp_unc = [table_madcuba[cond]['delta Tex/Te'].values[0]]*2
                table_species += [list(molecule.keys())[0]]
                
                for nonlte_molecule in nonlte_molecule_list:
                    if nonlte_molecule == table_species[-1]:
                        if num_sources < 1:
                            text_dens = (nonlte_molecule_list[nonlte_molecule]
                                         ['column density (/cm2)'])
                            text_temp = (nonlte_molecule_list[nonlte_molecule]
                                         ['kinetic temperature (K)'])
                        else:
                            text_dens = (nonlte_molecule_list[nonlte_molecule]
                                         ['column density (/cm2)'][s])
                            text_temp = (nonlte_molecule_list[nonlte_molecule]
                                         ['kinetic temperature (K)'][s])
                        if type(text_dens) is not type(None):
                            dens = rv.rich_value(text_dens)
                            dens_unc = [-10]*2 if dens.is_uplim else dens.unc
                            dens = dens.center
                        if type(text_temp) is not type(None):
                            temp = rv.rich_value(text_temp)
                            temp_unc = [-10]*2 if temp.is_uplim else temp.unc
                            temp = temp.center
                
            elif len(molecule) > 1:
                
                dens, dens_unc = 0, np.array([0., 0.])
                temp, temp_unc = 0, np.array([0., 0.])
                
                for variant in molecule:
                    
                    name = list(variant.values())[0]
                    cond = table_madcuba.Formula==name
                    dens_var = 10**table_madcuba[cond]['N/EM'].values[0]
                    delta = table_madcuba[cond]['delta N/EM'].values[0]
                    if len(false_lines) > 0:
                        false_detection = (name in false_lines[s]
                                           and false_lines[s][name] == 1)
                    else:
                        false_detection = False
                    if delta != -10 and not false_detection:
                        dens_var_unc = [10**delta]*2
                    else:
                        dens_var_unc = [-10]*2
                    if false_detection and correct_false_detections:
                        dens += \
                           10**delta * (sn_threshold
                                        - sigmas_false_lines[s][name])
                    temp_var = table_madcuba[cond]['Tex/Te'].values[0]
                    temp_var_unc = \
                        [table_madcuba[cond]['delta Tex/Te'].values[0]]*2
                            
                    table_species += [list(variant.keys())[0]]      
                    
                    for nonlte_molecule in nonlte_molecule_list:
                        if nonlte_molecule == table_species[-1]:
                            if num_sources < 1:
                                text_dens = (nonlte_molecule_list[nonlte_molecule]
                                             ['column density (/cm2)'])
                                text_temp = (nonlte_molecule_list[nonlte_molecule]
                                             ['kinetic temperature (K)'])
                            else:
                                text_dens = (nonlte_molecule_list[nonlte_molecule]
                                             ['column density (/cm2)'][s])
                                text_temp = (nonlte_molecule_list[nonlte_molecule]
                                             ['kinetic temperature (K)'][s])
                            if type(text_dens) is not type(None):
                                dens_var = rv.rich_value(text_dens)
                                dens_var_unc = ([-10]*2 if dens_var.is_uplim
                                                else dens_var.unc)
                                dens_var = dens_var.center
                            if type(text_temp) is not type(None):
                                temp_var = rv.rich_value(text_temp)
                                temp_var_unc = ([-10]*2 if temp_var.is_uplim
                                                else temp_var.unc)
                                temp_var = temp_var.center
                      
                    if temp_var_unc[0] < 0:
                        temp_var_unc = [0., 0.]
                      
                    dens += dens_var
                    dens_unc += np.array(dens_var_unc)**2
                    temp += temp_var
                    temp_unc += np.array(temp_var_unc)
                    
                    table_dens += [rv.RichValue(dens_var, dens_var_unc,
                                                is_uplim=dens_var_unc[0]<0,
                                                domain=[0,np.inf])]
                    table_temp += [rv.RichValue(temp_var, temp_var_unc,
                                                domain=[0,np.inf])]
                    table_abunds += [table_dens[-1] / reference_dens_s]
         
                dens_unc = np.sqrt(dens_unc)
                table_species += [' '.join(list(variant.keys())[0]
                                           .split(' ')[:-1])]
        
            if temp_unc[0] < 0:
                temp_unc = [0., 0.]
        
            dens_unc = np.array(dens_unc)
            temp_unc = np.array(temp_unc)
            temp /= len(molecule)
            temp_unc /= len(molecule)
            
            table_dens += [rv.RichValue(dens, dens_unc, is_uplim=dens_unc[0]<0,
                                        domain=[0,np.inf])]
            table_temp += [rv.RichValue(temp, temp_unc, domain=[0,np.inf])]
            table_abunds += [table_dens[-1] / reference_dens_s]
            
            abunds += [copy.copy(table_abunds[-1])]
            abunds_species += [table_species[-1]]
        
        abunds_table = rv.rich_dataframe({'species': table_species,
                                          'temperature': table_temp,
                                          'density': table_dens,
                                          'abundance': table_abunds}, num_sf=1)
        for column in factors_abunds:
            abunds_table[column] /= float(factors_abunds[column])
            abunds_table[column+'_unc'] /= float(factors_abunds[column])
        abunds_tables += [abunds_table]

        if export_abundances_list:
            abundances_df = rv.rich_dataframe({'molecule': abunds_species,
                                               'abundance': abunds}, num_sf=2)
            abundances_df.to_csv(output_list_s, index=False)
            source = input_madcuba_s.split('-')[0]
            print('Saved abundances for source {} in {}.'
                  .format(source, output_list_s))
        
if create_abundances_table or create_lines_table:
    
    if 'LaTeX template' in config['input files']:
        tables_template = config['input files']['LaTeX template']
        with open(tables_template, 'r') as file:
            table_ref = file.readlines()
    else:
        table_abunds_vars = []
        if create_abundances_table:
           for element in abunds_table_params:
               for row in element[1]:
                   table_abunds_vars += [' ' + row]
        table_abunds_vars = ' & '.join(table_abunds_vars) + ''
        table_abunds_vars = (table_abunds_vars
                             .replace('temperature', 'temperature (K)')
                             .replace('density', 'density (/cm$^2$)'))
        table_lines_vars = []
        if create_lines_table:
           for element in lines_table_params:
               for row in element[1]:
                   table_lines_vars += [' ' + row]
        table_lines_vars = ' & '.join(table_lines_vars)
        table_lines_vars = (table_lines_vars
                            .replace('frequency', 'frequency (MHz)')
                            .replace('snarea', 'S/N')
                            .replace('area', 'area (mK km/s)')
                            .replace('width', 'width (km/s)')
                            .replace('velocity', 'velocity (km/s)'))
        table_ref = ''
        table_ref = (default_tables_template
                           .replace('--tablelines_vars--', table_lines_vars)
                           .replace('--tableabunds_vars--', table_abunds_vars))
    output_file = config['output files']['LaTeX file']
    
    if create_abundances_table:
    
        for i,line in enumerate(abunds_table_params):
            abunds_table_params[i][0] = abunds_tables[line[0]-1]
             
        abunds_table = merge_tables(*abunds_table_params)
        
        for i, row in enumerate(scientific_notation['abundances table']):
            if row in ['density', 'abundance']:
                for j in range(len(abunds_tables) - 1):
                    scientific_notation['abundances table'] += [row + '_'*(j+1)]
        
        for column in abunds_table:
            if column in scientific_notation['abundances table']:
                for i in range(len(abunds_table[column])):
                    abunds_table.at[i,column].min_exp = 0

        abunds_table = rv.RichDataFrame(abunds_table)
    
        for i in range(len(abunds_table)):
            abunds_table.at[i,'species'] = \
                format_species_name(abunds_table['species'][i])

    if create_lines_table:
        
        for i,line in enumerate(lines_table_params):
            lines_table_params[i][0] = lines_tables[line[0]-1]
    
        lines_table = merge_tables(*lines_table_params)
        
        for i, row in enumerate(scientific_notation['lines table']):
            if row in ['area', 'width', 'velocity', 'sn_area', 'intensity']:
                for j in range(len(lines_tables) - 1):
                    scientific_notation['lines table'] += [row + '_'*(j+1)]

        for column in lines_table:
            if column in scientific_notation['lines table']:
                for i in range(len(lines_table[column])):
                    lines_table.at[i,column].min_exp = 0

        lines_table = rv.RichDataFrame(lines_table)
    
        name = lines_table['species'][0]
        for i in range(len(lines_table)-1):
            if lines_table['species'][i+1] == name:
                lines_table.at[i+1,'species'] = ''
            else:
                if lines_table['species'][i+1] != '':
                    name = lines_table['species'][i+1]
                    lines_table.at[i+1,'species'] = '*\hline*' + name
    
        for i in range(len(lines_table)):
            lines_table.at[i,'species'] = \
                format_species_name(lines_table['species'][i])
            lines_table.at[i,'transition'] = \
                format_transition_numbers(lines_table['transition'][i]) 
    
    tables = ''.join(table_ref)
    if create_abundances_table:
        tables = tables.replace('--tableabunds--', abunds_table.latex())
    if create_lines_table:
        tables = tables.replace('--tablelines--', lines_table.latex())
    
    tables = tables.replace('*\hline*', '\hline ')

    if 'non-LTE lines table' in config['input files']:
        nonlte_table = config['input files']['non-LTE lines table']
        table_name = list(nonlte_table.keys())[0]
        with open(table_name, 'r') as file:
            extra_table = file.readlines()
        extra_table = ''.join(extra_table)
        for molecule in nonlte_table[table_name]:
            molecule_fmt1 = \
                format_species_name(molecule, simplify_numbers=True)
            molecule_fmt2 = \
                format_species_name(molecule, simplify_numbers=False)
            if molecule_fmt1 in extra_table:
                molecule_name = molecule_fmt1
            elif molecule_fmt2 in extra_table:
                molecule_name = molecule_fmt2
            section_new = extra_table.split(molecule_name)[1].split('\\hline')[0]
            section_new = section_new.replace('\\end{tabular}\n', '')
            section_new = section_new.replace('\\end{table}\n', '')
            section_new = section_new.replace('\\end{landscape}', '')
            section_new = section_new.replace('\\end{document}\n', '')
            section_new = section_new.replace('\\par', '')
            if molecule_fmt1 in tables:
                molecule_name = molecule_fmt1
            elif molecule_fmt2 in tables:
                molecule_name = molecule_fmt2
            section_old = tables.split(molecule_name)[1].split('\\hline')[0]
            tables = tables.replace(section_old, section_new)
        
    with open(output_file, 'w') as file:
        file.writelines(tables)
    print('\nSaved tables in {}.'.format(output_file))

print()