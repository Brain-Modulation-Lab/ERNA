import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as subplots
import numpy as np
from time import sleep



# Function to create electrode layout
def create_electrode_layout(label, levels = 4, segments = 3):
    # Create subplots using plotly
    fig = subplots.make_subplots(rows=levels, cols=segments, shared_xaxes=True, shared_yaxes=True, 
                                 subplot_titles=[f"{label[i][j]}" for i in range(levels) for j in range(segments)])
    

        # Iterate over all subplots
    for i in range(levels):
        for j in range(segments):
            # Skip the specific subplots that should remain blank
            if (i == 0 and j == 0) or (i == 0 and j == 2) or (i == 3 and j == 0) or (i == 3 and j == 2):
                # These subplots will remain blank (no data, no axes)
                fig.update_xaxes(visible=False, row=i+1, col=j+1)
                fig.update_yaxes(visible=False, row=i+1, col=j+1)
            else:
                # Add a box around the subplot (using axis lines)
                fig.update_xaxes(showline=True, linewidth=2, linecolor='black', row=i+1, col=j+1)
                fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=i+1, col=j+1)
                # Set axis labels for populated subplots
                fig.update_xaxes(title_text="Time [ms]", row=i+1, col=j+1)
                fig.update_yaxes(title_text="ERNA [uV]", row=i+1, col=j+1)

    # Update layout for overall appearance
    fig.update_layout(
        height=900,
        title_text="Electrode Layout Real-Time Data",
        showlegend=False
    )
    
    return fig


@st.cache_data
def plot_voltage(fig, x, y, xlim, ylim, winpeak):
    if fig:
        figLH.update_xaxes(range=xlim)
        figLH.update_yaxes(range=ylim)
        if x and y:
            trace = go.Scatter(
                x=x, y=y, mode='lines'
            )
            fig.add_trace(trace, row=i+1, col=j+1)
        if winpeak:
            # Iterate over all subplots
            for i in range(levels):
                for j in range(segments):
                    # Skip the specific subplots that should remain blank
                    if (i == 0 and j == 0) or (i == 0 and j == 2) or (i == 3 and j == 0) or (i == 3 and j == 2):
                        # These subplots will remain blank (no data, no axes)
                        fig.add_vline(x=winpeak, line=dict(color="red", width=2),row=i+1, col=j+1)
    
        st.plotly_chart(fig)                    
    else:
        print("Create electrode layout first!")




