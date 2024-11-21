import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as subplots
import numpy as np
from time import sleep


def generate_real_time_data():
    return np.random.rand(10)  # Simulate 10 data points for example


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
                # Populate with random data
                data = generate_real_time_data()
                trace = go.Scatter(
                    x=np.arange(len(data)), y=data, mode='lines', name=f"Data {i+1},{j+1}"
                )
                fig.add_trace(trace, row=i+1, col=j+1)
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
        showlegend=False,
        plot_bgcolor='white',  # White background for the plot area
        paper_bgcolor='gray',  # White background for the paper (the area outside the plot)
        font=dict(color='black'),  # Set font color to black for better contrast
        xaxis=dict(
            showgrid=True, gridcolor='lightgrey', zeroline=False, linecolor='black'
        ),
        yaxis=dict(
            showgrid=True, gridcolor='lightgrey', zeroline=False, linecolor='black'
        )
    )
    
    return fig

        

