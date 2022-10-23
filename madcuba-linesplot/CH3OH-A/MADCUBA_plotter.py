#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pandas as pd
import numpy as np

from pathlib import Path
import os
import glob
from copy import deepcopy

import warnings
import string



def MADCUBA_plot(m, n, fig_name='Figure', fig_format='png',
                 x_limits=['NA', 'NA'], y_limits=['NA', 'NA'],
                 x_label='x', y_label='y', drawstyle='histogram', 
                 spec_color='k', linewidth=0.8, linestyle='-',
                 fit_color=['red'], fit_linewidth=0.8, fit_linestyle='-', 
                 labelsize=12, labelfont='Arial', molfontize=6, moleculefont='courier new',
                 panel_naming='letters', panelfont='Arial', panfontsize=8):
    
    #++++++++++++++++ Function Parameters Description ++++++++++++++++#

    """
    m               = number of rows           - int/float
    n               = number of columns        - int/float
    fig_name        = name to save figure      - string
                    # Default: 'Figure'
    fig_format      = output figure format     - string
                      png, pdf, ps, eps or svg
                    # Default: 'png'  
    x_limits        = min and max for x axis   - list of int/floats
                    # Default: ['NA', 'NA'] (auto. sets  min and max from all spec)
    y_limits        = min and max for y axis   - list of int/floats
                    # Default: ['NA', 'NA'] (auto. sets  min and max from all spec)
    x_label         = label for x axis         - string
                    # Default: 'x'
    y_label         = label for y axis         - string
                    # Default: 'y'
    drawstyle       = 'histogram'              - step-type spectra
                      'default'                - line-type spectra
                    # Default: 'histogram'
    spec_color      = color for the spectra    - string
                    # Default: 'k'
    fit_color       = olors for the LTE fit    - list of strings
                    # Default: ['red']
    linewidth       = plotted  linewidth       - float
                    # Default: 0.8
    linesyle        = plotted  linestyle       - string
                    # Default: '-'
    fit_linewidth   = plotted fit linewidth    - float
                    # Default: 0.8
    fit_linesyle    = plotted fit linestyle    - string
                    # Default: '-'
    labelsize       = label font size          - float
                    # Default: 12
    labelfont       = label font               - string
                    # Default: 'Arial'
    molfontize      = tick label sizes         - float
                    # Default: 6
    moleculefont    = molecule label font      - string
                    # Default: 'courier new'
    panel_naming    = name panels              - string
                    'letters', 'numbers', 'NA' 
                    # Default: 'letters'
    panelfont       = name panel font          - string
                    # Default: 'Arial'
    panfontsize     = name panel font size     - float
                    # Default: 8
    """
    
    matplotlib.rcParams['pdf.fonttype'] = 42 # Using TrueType fonts
    matplotlib.rcParams['ps.fonttype']  = 42 # Using TrueType fonts

    # Working directory
    workdir =  os.getcwd()
    
    # Paths
    data_files = sorted(glob.glob(workdir+'/data/*'))
    spec_files = sorted(glob.glob(workdir+'/spectroscopy/*'))
    
    # Number of existing specs
    spec_number = len(data_files)
    
    # Setting draw style
    if drawstyle=='histogram':
        ds='steps-mid'
    else:
        ds='default'
    
    # Figure Size
    size_rat = float(n)/float(m)
    size_x = 10.*size_rat
    size_y = 10.
        
    fig = plt.figure(figsize=(size_x, size_y))
    gs1 = gridspec.GridSpec(m, n)    
    gs1.update(wspace = 0.0, hspace=0.0, top=0.95, bottom = 0.05)
    
    high_label = 0.82
    low_label = 0.6
    
    
    
    axis = []
    axis_ind = []
    # Generating specified number of axis
    #for i in range(spec_number):
    for i in range(m*n):
        row = (i // n)
        col = i % n
        axis.append(fig.add_subplot(gs1[i]))#(gs1[row, col]))#(gs1[i]))
            
    
    ind = 0
    axis_ind = []
    #for i  in range(m):
    for i  in range(m):
        axis_ind.append([])
        for j in range(n):
            axis_ind[i].append(ind)
            ind += 1
        
    y_max_list = []
    y_min_list = []
    x_max_list = []
    x_min_list = []
    panel_names = []
    
    
    
    # Checking if grid is smaller than number of panels from SLIM
    if m*n < spec_number:
        raise IOError('Entered grid (%sx%x) smaller than number of panels: %s' %(m,n,len(glob.glob(workdir+'/data/*'))))
    
    # Reading Data for plotting
    for j in range(m*n):
        if j < spec_number:
            if Path(data_files[j]).is_file():  # Checking if file exists
                panel_names.append(data_files[j].split('/')[-1])
                # Reading asciis from SLIM
                data = pd.read_csv(data_files[j], delim_whitespace= True, header=None)
                data_ncolumns = data.shape[1]
                data_cols = ['vel', 'int']
                fit_ncols = data_ncolumns - 2
                
                # Raise warning when fit_color list is smaller than the number of fits
                if len(fit_color) < fit_ncols:
                    warnings.warn('Fit color list lenght is smaller than number of fits plotted,'
                                  ' setting default colors.')
                    cmap = matplotlib.cm.get_cmap('tab10')
                    lin = np.linspace(0, 1, fit_ncols)
                    
                    for r, rgb in enumerate(lin):
                        fit_color.append(cmap(rgb)) 
                    
                
                # Appending if there is more than one fit available
                if data_ncolumns > 2:
                    for f in range(data_ncolumns-2):
                        data_cols.append('fit_'+str(f+1))
                data.columns = data_cols
            
            
                # y-axis Max and min for every panel
                y_max_list.append(np.nanmax(data['int'].tolist()))
                y_min_list.append(np.nanmin(data['int'].tolist()))
            
                # x-axis Max and min for every panel
                x_max_list.append(np.nanmax(data['vel'].tolist()))
                x_min_list.append(np.nanmin(data['vel'].tolist()))
                
                # Plotting spectrum
                axis[j].plot(data['vel'], data['int'], linewidth=linewidth,  linestyle=linestyle, drawstyle=ds, color=spec_color)
                # Plotting LTE fit
                for ff in range(fit_ncols):
                    axis[j].plot(data['vel'], data['fit_'+str(ff+1)], color=fit_color[ff], linewidth=fit_linewidth, linestyle=fit_linestyle)
                    
        else: # If there are less datafiles than panels in grid mxn (empty panels)
            panel_names.append(str(j))
            data = pd.DataFrame(columns=['vel', 'int', 'fit'])
            y_max_list.append(np.nan)
            y_min_list.append(np.nan)
            x_max_list.append(np.nan)
            x_min_list.append(np.nan)
            
        
        
    # Overall max and min to set same axis limits in all panels
    y_total_max = np.nanmax(y_max_list)
    y_total_min = np.nanmin(y_min_list)
    x_total_max = np.nanmax(x_max_list)
    x_total_min = np.nanmin(x_min_list)    
    if x_limits != ['NA', 'NA']: # Si no se especifican limites python coge el minimo y maximo de cada expectro
            x_total_min = x_limits[0]
            x_total_max = x_limits[1]
        
    x_len = x_total_max - x_total_min
    label_min_sep = x_len*5.0/100.
    label_same_sep = x_len*1.0/100.
    
    # Annotating line labels
    line_annotation(m, n, spec_files, panel_names, y_total_min, y_total_max, y_limits,
                   label_same_sep, label_min_sep,
                   x_total_min, x_total_max, low_label, high_label, axis, molfontize, moleculefont
                   )

    #### Panel parameters
    # Left Figures
    left_ind = []
    for i in range(m):
        left_ind.append(axis_ind[i][0])
    # Bottom Figures
    bottom_figures = axis_ind[-1]
    
    # Pane names
    if panel_naming == 'letters':
        panenames_1 = list(string.ascii_lowercase)[0:len(axis)+1]
        panenames_2 = []
        panenames_3 = []
        if m*n > 26:
            panenames_2 = [''.join(i).strip() for i in zip(['a']*len(panenames_1), panenames_1)]
        elif m*n > 26*2:
            panenames_3 = [''.join(i).strip() for i in zip(['b']*len(panenames_1), panenames_1)]
        elif m*n > 26*3:
            raise IOError('Too many plots (>78): %s' %(m*n))
        panenames = panenames_1+panenames_2+panenames_3
    elif panel_naming == 'numbers':
        panenames =  [str(x) for x in range(1, len(axis) + 1)]
    else:
        panenames = ['']*len(axis)
    for i, ax in enumerate(axis):
        ax.tick_params(axis='both', which='both', direction='in')
        ax.minorticks_on()
        ax.xaxis.set_tick_params(which='both', top ='on')
        ax.yaxis.set_tick_params(which='both', right='on', labelright='off')
        
        # Labeling each panel:
        axis[i].text(0.075, 0.95, panenames[i]+')',
                            horizontalalignment='left',
                            verticalalignment='top',
                            fontsize=panfontsize, fontname=panelfont,
                            transform=axis[i].transAxes)
        
        # X axis limits
        if x_limits != ['NA', 'NA']: 
            ax.set_xlim(x_limits)    
        else:
            # If no axes limits are given, setting it to total max and min from
            # all spec
            ax.set_xlim([x_total_min, x_total_max]) # Todos con los mismos limites
            
        # Y axis limits
        if y_limits != ['NA', 'NA']:
            ax.set_ylim(y_limits)    
        else:
            # If no axes limits are given, setting it to total max and min from
            # all spec
            ax.set_ylim([y_total_min, y_total_max+y_total_max*0.4]) 
        
        # Only adding y axis labels to left axes
        ax.tick_params(axis='both', which='major', labelsize=labelsize)
        if i in left_ind:
            if y_label != 'NA':
                ax.set_ylabel(y_label, fontsize=labelsize, fontname=labelfont)
        else:
            ax.set_yticklabels([])
        # Only adding x axis labels to bottom axes
        if i in bottom_figures:
            if x_label != 'NA':
                ax.set_xlabel(x_label, fontsize=labelsize, fontname=labelfont)
        else:
            ax.set_xticklabels([])
    fig.savefig(workdir+'/'+fig_name+'.'+fig_format, bbox_inches='tight', transparent=True, dpi=400, format=fig_format)
    plt.close()
    

def line_annotation(m, n, spec_files, panel_names, y_total_min, y_total_max, y_limits,
                   label_same_sep, label_min_sep,
                   x_total_min, x_total_max, low_label, high_label, axis, molfontize, moleculefont
                   ):
    """
    Esto es bastante lio, se puede hacer que se escriban sin mas aunque solapen
    """
    # Annotating line labels
    for j in range(m*n):
        # Plotting line position and names
        for k, spec_filename in enumerate(spec_files):
            if Path(spec_filename).is_file(): 
                spec_name = spec_filename.split('/')[-1]
                if panel_names[j] == spec_name:
                    
                    spec = pd.read_csv(spec_filename, delim_whitespace= True, header=None)
                    spec.columns = ['vel', 'label', 'transition', 'freq']
                    spec = spec.sort_values(['vel'], ascending = True).reset_index(drop=True)

                    # Label position at 80% of panel height
                    if y_limits != ['NA', 'NA']: 
                        y_label_pos = np.nanmax(y_limits)
                    else:
                        y_label_pos = y_total_max
                    subset = []
                    same_sub = []
                    difs_name = []
                    for l, line in spec.iterrows():
                        spec['dif_'+str(l)] = np.abs(spec['vel'] - line['vel'])
                        difs_name.append('dif_'+str(l))
                        # x-Difference 1%<diff<3% 
                    for d, dif_col in enumerate(difs_name):
                        # x-Difference diff<3% 
                        sub =spec[(spec[dif_col]<2*label_same_sep)]
                        col_list = ['vel', 'label', 'transition', 'freq'] + [dif_col]
                        subset.append(sub[col_list])
                        sub2 = spec[(spec[dif_col]<label_same_sep)]
                        if len(sub2)>1:
                            same_sub.append(sub2[col_list])

                    tam = len(subset)   
                    col_list = ['vel', 'label', 'transition', 'freq']
                    subset.sort(key=len)#, reverse = True)
                    ff_d=[]
                    count = 0
                    subset2 = deepcopy(subset)
                    for p, df in enumerate(subset):
                        if p > 0:
                            prev_ = subset[p-1][col_list]
                        else:
                            prev_ = pd.DataFrame(columns=col_list)
                        if p < (tam-1):
                            next_ = subset[p+1][col_list]
                        else:
                            next_ = pd.DataFrame(columns=col_list)
                        df_m = deepcopy(df[col_list])
                        for q, df2 in enumerate(subset2):
                            if p!=q:
                                # Add dfs together if they share any element (close lines)
                                if any(i in df_m.index.tolist() for i in df2.index.tolist()):
                                        df_m = df_m.append(df2[col_list], ignore_index=False)
                                        df_m.drop_duplicates(inplace=True)
                                        ff_d.append(df_m.sort_index())
                        if len(df)==1:
                            ff_d.append(df[col_list])
                             
                    
                    rep_list = []
                    df_list =[]
                    for p, df in enumerate(ff_d):
                        c = [i for i, e in enumerate(ff_d) if e.equals(df)]
                        if c[0] in rep_list:
                            continue
                        else:
                            rep_list.append(c[0])
                            df_list.append(ff_d[c[0]])
                            
                    count = 0
                    count2 = 0
                    total = 1
                    
                    line_list = []
                    for s, sub_df in enumerate(df_list):
                        subdf = sub_df.reset_index(drop=True)
                        for k, line in subdf.iterrows():
                            if (line['vel'] > float(x_total_min)) & (line['vel'] < float(x_total_max)):
                                line_list.append(line['vel'])
                    for s, sub_df in enumerate(df_list):
                        lab_pos_list = np.array([low_label, high_label])
                        subdf = sub_df.reset_index(drop=True)
                        l = 0
                        tam = len(subdf)
                        if tam == 1:
                            if total % 2 == 0:
                                label_pos = lab_pos_list[0]
                                min_lab_pos = label_pos
                                            
                            else:
                                label_pos = lab_pos_list[1]
                            for k, line in subdf.iterrows():
                                if (line['vel'] > float(x_total_min)) & (line['vel'] < float(x_total_max)):
                                    axis[j].vlines(line['vel'], ymin=y_total_min, ymax=y_label_pos*label_pos, color='k', linestyle='--', lw=0.5)
                                    axis[j].text(line['vel'], y_label_pos*(label_pos+0.01), line['label'], ha='center', va='bottom',rotation='vertical', backgroundcolor='none', fontsize=molfontize, fontname=moleculefont)
                                    total = total +1
                        else:
                            if 2.0*subdf['vel'].std() > label_min_sep:
                                for k, line in subdf.iterrows():
                                    if tam >= 2:
                                        if total % 2 == 0:
                                            label_pos = low_label
                                        else:
                                            label_pos = high_label
                                    else:
                                        label_pos = high_label
                                    if (line['vel'] > float(x_total_min)) & (line['vel'] < float(x_total_max)):
                                        axis[j].vlines(line['vel'], ymin=y_total_min, ymax=y_label_pos*label_pos, color='k', linestyle='--', lw=0.5)
                                        axis[j].text(line['vel'], y_label_pos*(label_pos+0.01), line['label'], ha='center', va='bottom',rotation='vertical', backgroundcolor='none', fontsize=molfontize, fontname=moleculefont)
                                        total = total +1 
                                    l+=1
                            else:
                                
                                if tam > 2:
                                    if (tam%2) == 1:
                                        sep_max = (tam / 2)*label_same_sep*2.5
                                        mid = subdf['vel'].loc[tam/2]
                                        if (mid > float(x_total_min)) & (mid < float(x_total_max)):
                                            axis[j].vlines(mid, ymin=y_total_min, ymax=y_label_pos*(low_label), color='k', linestyle='--', lw=0.5)
                                            linspace = np.linspace(mid-sep_max, mid+sep_max, tam, endpoint=True)
                                    
                                    else:
                                        sep_max = (tam / 2)*label_same_sep*1.5
                                        mid = np.mean([subdf['vel'].loc[tam/2], subdf['vel'].loc[tam/2 -1]])
                                        if (mid > float(x_total_min)) & (mid < float(x_total_max)):
                                            axis[j].vlines(mid, ymin=y_total_min, ymax=y_label_pos*(low_label-0.05), color='k', linestyle='--', lw=0.5)
                                            linspace = np.linspace(mid-sep_max, mid+sep_max, tam, endpoint=True)
                                            
                                    for k, line in subdf.iterrows():
                                        if total % 2 == 0:
                                            label_pos = lab_pos_list[0]
                                            min_lab_pos = label_pos
                                            
                                        else:
                                            label_pos = lab_pos_list[1]
                                            axis[j].vlines(linspace[k], ymin=y_label_pos*(low_label), ymax=y_label_pos*high_label, color='k', linestyle='--', lw=0.5)
                                            min_lab_pos = low_label 
                                        if (line['vel'] > float(x_total_min)) & (line['vel'] < float(x_total_max)):
                                            axis[j].text(linspace[k], y_label_pos*(label_pos+0.01), line['label'], ha='center', va='bottom',rotation='vertical', backgroundcolor='none', fontsize=molfontize, fontname=moleculefont)
                                            if linspace[k]<mid:
                                                axis[j].plot([linspace[k], mid], [y_label_pos*min_lab_pos, y_label_pos*(low_label - 0.05)], color='k', linestyle='--', lw=0.5)
                                            elif linspace[k]>mid:
                                                axis[j].plot([mid, linspace[k]], [y_label_pos*(low_label - 0.05), y_label_pos*min_lab_pos], color='k', linestyle='--', lw=0.5)
                                            total = total +1 
                                            count += 1
                                elif tam == 2:
                                    if len(line_list) > 1:
                                        prev_ = line_list[total-2]
                                    else:
                                        prev_ = 0
                                    sep_max = (tam / 2)*label_same_sep*1.5
                                    mid = np.mean([subdf['vel'].loc[tam/2], subdf['vel'].loc[tam/2 -1]])
                                    if (mid > float(x_total_min)) & (mid < float(x_total_max)):
                                        linspace = np.linspace(mid-sep_max, mid+sep_max, tam, endpoint=True)         
                                    any_dif = []
                                    for pp, prev in enumerate(line_list):
                                        if np.abs(prev - np.min(linspace))<label_min_sep:
                                            any_dif.append(True)
                                        else:
                                            any_dif.append(False)
                                    if True in any_dif:
                                        pos_text = [mid, mid + sep_max]
                                        shift = True
                                    else:
                                        pos_text = linspace
                                        shift = False
                                        
                                    if shift == True:
                                        for k, line in subdf.iterrows():
                                                #pos_text = [mid, mid + sep_max]
                                            
                                            if (line['vel'] > float(x_total_min)) & (line['vel'] < float(x_total_max)):
                                                if total%2 == 0:
                                                    label_pos = lab_pos_list[1]
                                                    label_pos_oth = lab_pos_list[0]
                                                    #pos_text = [mid - sep_max, mid]
                                                else:
                                                    label_pos_oth = lab_pos_list[1]
                                                    label_pos = lab_pos_list[0]
                                                axis[j].text(pos_text[k], y_label_pos*(label_pos+0.01), line['label'], ha='center', va='bottom',rotation='vertical', backgroundcolor='none', fontsize=molfontize, fontname=moleculefont)
                                            
                                                if pos_text[k]<mid:
                                                    axis[j].plot([pos_text[k],mid], [y_label_pos*label_pos, y_label_pos*(high_label - 0.05)], color='k', linestyle='--', lw=0.5)
                                                elif pos_text[k]>mid:
                                                    axis[j].plot([mid, pos_text[k]], [y_label_pos*(label_pos - 0.05), y_label_pos*label_pos], color='k', linestyle='--', lw=0.5)
                                                elif pos_text[k]==mid:
                                                    axis[j].vlines(mid, ymin=y_total_min, ymax=y_label_pos*(label_pos), color='k', linestyle='--', lw=0.5)
    
                                                #    axis[j].plot([pos_text[k],], [y_label_pos*(high_label - 0.05), y_label_pos*label_pos], color='k', linestyle='--', lw=0.5)
                                                
                                                total = total +1 
                                    else:
                                        axis[j].vlines(mid, ymin=y_total_min, ymax=y_label_pos*(label_pos-0.05), color='k', linestyle='--', lw=0.5)
                                        for k, line in subdf.iterrows():
                                            if (line['vel'] > float(x_total_min)) & (line['vel'] < float(x_total_max)):
                                                label_pos = high_label
                                                axis[j].text(linspace[k], y_label_pos*(label_pos+0.01), line['label'], ha='center', va='bottom',rotation='vertical', backgroundcolor='none', fontsize=molfontize, fontname=moleculefont)
                                                if pos_text[k]<mid:
                                                    axis[j].plot([linspace[k],mid], [y_label_pos*label_pos, y_label_pos*(high_label - 0.05)], color='k', linestyle='--', lw=0.5)
                                                elif pos_text[k]>mid:
                                                    axis[j].plot([mid, linspace[k]], [y_label_pos*(label_pos - 0.05), y_label_pos*label_pos], color='k', linestyle='--', lw=0.5)
                                                
                        if count % 2 == 1:
                            count2 = count2 + 2
                        else:
                            count2 = count2 + 1