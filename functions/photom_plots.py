import datetime

import matplotlib
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import colors


def plot_heatmap(Stim, Signal, AlignedTo, TrialsData, DS_rate):
    # create figure
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(1, 1, 1)
    # select trials for this stimulus and transpose
    HeatMapData = TrialsData[AlignedTo+'_'+Signal].xs(Stim, level = 2, axis = 1).T
    # center color scale around 0
    divnorm = colors.TwoSlopeNorm(vmin = HeatMapData.min().min(), vcenter = 0, vmax = HeatMapData.max().max())
    # plot heatmap
    im = ax.imshow(HeatMapData,aspect='auto',interpolation='none',norm = divnorm,cmap='BrBG')
    # add x ticks & labels
    ax.set_xticks(range(0,len(HeatMapData.columns.tolist()),180*DS_rate))
    ax.set_xticklabels([str(int(i/60)) for i in HeatMapData.columns.tolist()[0:len(HeatMapData.columns.tolist()):180*DS_rate]])
    ax.set_yticks([0,len(HeatMapData.index)-1])
    ax.set_yticklabels([1,len(HeatMapData.index)])
    # add line at t=0
    ax.plot([
        np.where(HeatMapData.columns == 0)[0][0],
        np.where(HeatMapData.columns == 0)[0][0]],
        [-0.5,len(HeatMapData)-0.5],
        color='y'
    )
    # axis properties
    ax.set(ylabel = 'trial #', xlabel = 'time (mins)', title = Signal + ' ' + AlignedTo + 'Aligned: ' + Stim)
    # colorbar
    cbar = plt.colorbar(im,ax=ax)
    cbar.set_label(Signal)

    return fig, ax

def plot_mean_over_trials(Stims, Signal, AlignedTo, my_colors, TrialData, DS):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    TrialData = TrialData[AlignedTo+'_'+Signal]
    StimN = 0
    for Stim in Stims:
        # Select trials for this stimulus
        StimTrials = TrialData.xs(Stim, level = 2, axis = 1)
        # Calculate mean over these trials and downsample
        StimMean = np.mean(StimTrials, 1)[::DS]
        # Calculate SEM and downsample
        StimSEM = np.std(StimTrials,1,ddof=1) / np.sqrt(len(StimTrials.columns))
        StimSEM = StimSEM[::DS]
        # Shade in SEM
        ax.fill_between(np.array(StimMean.index,dtype=float), np.add(StimMean,StimSEM).tolist(),
                            np.subtract(StimMean,StimSEM).tolist(), color=my_colors[StimN],
                            alpha = 0.4)
        # Plot mean trace
        ax.plot(StimMean, label = Stim, color=my_colors[StimN])
        StimN = StimN + 1
    # Plot line at t=0
    yl = list(ax.get_ylim())
    ax.plot([0,0],yl,color='black')
    ax.set_ylim(yl)
    ax.legend(loc="lower right")
    ax.set_xlabel('time (secs)')
    ax.set_ylabel(Signal)
    ax.set_title('Mean over trials - aligned to '+AlignedTo)
    # Save this figure
    #fig.savefig(Signal+' '+AlignedTo+'Aligned' + ', '.join(StimsToPlot)+': MeanOverTrials.png')

    return fig, ax

def make_boxplot(df, my_colors, connecting_lines=True):
    fig, ax = plt.subplots(1,1,figsize=(2,6))
    df.boxplot(ax=ax, widths=(0.5,)*len(df.columns), patch_artist=True)
    ax.grid(False)
    #ax.xaxis.set_visible(False)
    for pos in ['right', 'top', 'bottom']:
        ax.spines[pos].set_visible(False)
    _ = ax.set_ylabel(u'Mean zdF/F\u2080 from 0 to 5min')

    for i in range(len(my_colors)):
        ax.findobj(matplotlib.patches.Patch)[i].set_color(my_colors[i])
        j = i*6
        for k in range(j,j+4):
            ax.findobj(matplotlib.lines.Line2D)[k].set_color('black')
        ax.findobj(matplotlib.lines.Line2D)[k+1].set_color('black')

    if connecting_lines==True:
        ax.plot(ax.get_xticks(), df.transpose(), color='black', alpha=0.2)
    
    return fig, ax

def plot_mean_over_mice(Stims, Signal, AlignedTo, MouseData, my_colors, DS):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    MouseData = MouseData[Signal][AlignedTo]
    StimN = 0
    for Stim in Stims:
        # Select trials for this stimulus
        StimTrials = MouseData.xs(Stim, level = 1, axis = 1)
        # Calculate mean over these trials and downsample
        StimMean = np.mean(StimTrials, 1)[::DS]
        # Calculate SEM and downsample
        StimSEM = np.std(StimTrials,1,ddof=1) / np.sqrt(len(StimTrials.columns))
        StimSEM = StimSEM[::DS]
        # Shade in SEM
        ax.fill_between(np.array(StimMean.index,dtype=float), np.add(StimMean,StimSEM).tolist(),
                            np.subtract(StimMean,StimSEM).tolist(), color=my_colors[StimN],
                            alpha = 0.4)
        # Plot mean trace
        ax.plot(StimMean, label = Stim, color=my_colors[StimN])
        StimN = StimN + 1
    # Plot line at t=0
    yl = list(ax.get_ylim())
    ax.axvline(0, color='black')#ax.plot([0,0],yl,color='black')
    ax.set_ylim(yl)
    ax.legend(loc="lower right")
    ax.set_xlabel('time (secs)')
    ax.set_ylabel(Signal)
    ax.set_title('Mean over mice - aligned to '+AlignedTo)
    # Save this figure
    #fig.savefig(Signal+' '+AlignedTo+'Aligned' + ', '.join(StimsToPlot)+': MeanOverTrials.png')

    return fig, ax

def align_to_event(event, event_pretime, data, mouse_data):
    # Convert pretime to seconds
    event_pretime = event_pretime*60
    # Get number of samples from drop to event for each trial
    RelTimes = [np.nan if type(data[event+'Times'][i])==list else
                data[event+'Times'][i] - data['DropTimes'][i] 
                for i in list(range(len(data[event+'Times'])))]
    # Convert this into time given sampling rate
    data[event+'Inds'] = [round(x*data['GCaMP_corrected'].index[1],2) for x in RelTimes]

    # Find lowest length of time from event to end of trial
    # We will plot this length of time after event for every trial
    event_posttime = np.nanmin([round(np.amax(list(data['Drop_dFF0'].index))-x,2) for x in data[event+'Inds']])-1
    
    # Initialize dataframe for event-aligned traces
    event_ind = list(np.arange(0-event_pretime, event_posttime, data['GCaMP_corrected'].index[1]))
    event_ind = [round(x,2) for x in event_ind]
    my_cols = pd.MultiIndex(levels = [[],[],[]],
                            codes = [[],[],[]],
                            names = [u'MouseID',u'TrialID',u'DropWhat']
                            )
    data[event+'_dFF0'] = pd.DataFrame(index = event_ind, columns = my_cols)
    data[event+'_ZS'] = pd.DataFrame(index = event_ind, columns = my_cols)

    # Loop through trials
    for i in range(len(data[event+'Inds'])):
        EventInd = data[event+'Inds'][i]
        #get the data for this trial
        ThisTrial_dFF0 = data['Drop_dFF0'].iloc[:,i]
        ThisTrial_ZS = data['Drop_ZS'].iloc[:,i]
        cols1 = data['Drop_dFF0'].columns[i]
        # Align this trace to the event time
        AlignedTrace_dFF0 = ThisTrial_dFF0[round(EventInd-event_pretime,2):round(EventInd+event_posttime,2)]
        AlignedTrace_ZS = ThisTrial_ZS[round(EventInd-event_pretime,2):round(EventInd+event_posttime,2)]
        # Make indices relative to event time
        event_ind = [round(x-EventInd,2) for x in list(AlignedTrace_dFF0.index)]
        # Get column info (MouseID, TrialID, DropWhat)
        coldata = list(data['Drop_dFF0'].columns[i])
        cols1 = pd.MultiIndex.from_arrays([[coldata[0]],[coldata[1]],[coldata[2]]])
        # Create dataframe
        AlignedTrace_dFF0 = pd.DataFrame(list(AlignedTrace_dFF0), index = event_ind, columns = cols1)
        AlignedTrace_ZS = pd.DataFrame(list(AlignedTrace_ZS), index = event_ind, columns = cols1)
        # Add to data
        data[event+'_dFF0'] = pd.concat([data[event+'_dFF0'], AlignedTrace_dFF0], axis=1)
        data[event+'_ZS'] = pd.concat([data[event+'_ZS'], AlignedTrace_ZS], axis=1)
    
    # Now take mean over trials for each mouse, for each stimulus type
    # Initialize dataframe to contain mouse means
    my_cols = pd.MultiIndex(levels = [[],[]], codes = [[],[]],
                            names = [u'Mouse',u'Stim'])
    mouse_data['dFF0'][event] = pd.DataFrame(index = event_ind, columns = my_cols)
    mouse_data['ZS'][event] = pd.DataFrame(index = event_ind, columns = my_cols)

    for Mouse in list(set(data[event+'_dFF0'].columns.get_level_values(0))): #loop through mice
        # get trials from this mouse
        MouseTrials_dFF0 = data[event+'_dFF0'].xs(Mouse, level=0, axis=1)
        MouseTrials_ZS = data[event+'_ZS'].xs(Mouse, level=0, axis=1)
        # loop through unique stimulus IDs for this mouse
        for Stim in list(set(MouseTrials_dFF0.columns.get_level_values(1))):
            # calculate mean over trials for this mouse
            ThisMouseMean_dFF0 = pd.DataFrame(np.mean(MouseTrials_dFF0.xs(Stim, level=1, axis=1),1), 
                                                index=list(MouseTrials_dFF0.index), columns=pd.MultiIndex.from_arrays([[Mouse],[Stim]]))
            ThisMouseMean_ZS = pd.DataFrame(np.mean(MouseTrials_ZS.xs(Stim, level=1, axis=1),1),
                                                index=list(MouseTrials_ZS.index), columns=pd.MultiIndex.from_arrays([[Mouse],[Stim]]))
            # add to dataframe
            mouse_data['dFF0'][event] = pd.concat([mouse_data['dFF0'][event],ThisMouseMean_dFF0], axis=1)
            mouse_data['ZS'][event] = pd.concat([mouse_data['ZS'][event],ThisMouseMean_ZS], axis=1)

    return data, mouse_data

def plot_sem_ax(mean_trace, sem_trace, ConditionsToPlot, ColorsToUse, ax):
    mean_trace.plot(color=ColorsToUse, linewidth=3, ax=ax)
    # Add SEM
    i=0
    for Stim in ConditionsToPlot:
        plt.fill_between(sem_trace.index,
                    np.add(mean_trace[Stim],sem_trace[Stim]).tolist(),
                    np.subtract(mean_trace[Stim],sem_trace[Stim]).tolist(),
                    color = ColorsToUse[i], alpha=0.2)
        i=i+1
    # Shade in lights-off period (ZT12-ZT0) for each day
    ylim = ax.get_ylim()
    for j in list(range(1,len(list(set([i.day for i in list(mean_trace.index)]))))):
        plt.fill_between([datetime.datetime(1,1,j,12,0,0),datetime.datetime(1,1,j+1,0,0,0)],[ylim[0],ylim[0]],[ylim[1],ylim[1]],
        color='darkgray', alpha=0.2)
    _ = ax.set_ylim(ylim)

    # Set axis labels, title
    ax.legend(loc="upper right", fontsize="large")

    # change x tick labels to just show hours (not days)
    date_form = DateFormatter("%H")
    ax.xaxis.set_major_formatter(date_form)
    ax.set_xticks(ax.get_xticks()) # to suppress FixedLocator warning
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')

    # hide frame
    for pos in ['right', 'top', 'left']:
        ax.spines[pos].set_visible(False)

    return ax

def plot_mean_sem(mean_trace, sem_trace, ConditionsToPlot, ColorsToUse):
    fig,ax = plt.subplots(1,1)
    ax = plot_sem_ax(mean_trace, sem_trace, ConditionsToPlot, ColorsToUse, ax)

    return fig, ax

def make_hboxplot(df, my_colors, connecting_lines=True):
    fig, ax = plt.subplots(1,1,figsize=(8,4))
    df.boxplot(ax=ax, widths=(0.5,)*len(df.columns), patch_artist=True, vert=False)
    ax.grid(False)
    #ax.xaxis.set_visible(False)
    for pos in ['right', 'top', 'bottom']:
        ax.spines[pos].set_visible(False)
    
    for i in range(len(my_colors)):
        ax.findobj(matplotlib.patches.Patch)[i].set_color(my_colors[i])
        j = i*6
        for k in range(j,j+4):
            ax.findobj(matplotlib.lines.Line2D)[k].set_color('black')
        ax.findobj(matplotlib.lines.Line2D)[k+1].set_color('black')

    if connecting_lines==True:
        ax.plot(df.transpose(), ax.get_yticks(), color='black', alpha=0.2)
    
    return fig, ax
