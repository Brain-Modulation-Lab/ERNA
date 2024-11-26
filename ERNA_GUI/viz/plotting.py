import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import plotly.graph_objects as go
from pathlib import Path
import os
import plotly.io as pio
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express as px
from ERNA_GUI.api.utility import (
    fetch_IBImatrix, fetch_IPImatrix, load_config
)

# get base dir of package
current_script = Path(__file__)
BASE_DIR = current_script.parent.parent.parent






# Function to generate distinctive colors
def linspecer(N):
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    return colors

# show recordings
def show_recordings(fig, coords, radius = 1):
    # Plot the recording location
    for coord in coords:
        x,y,z = create_spheres(coord, radius)
        if coord[0] > 0:
            x  = - x # flip from RH to LH
        fig.add_trace(go.Mesh3d(
            x=x.flatten(), 
            y=y.flatten(), 
            z=z.flatten(), color='red', opacity=0.50, alphahull=0
            )
        )
    fig.update_traces(lighting=dict(ambient=0.5, specular=1.0))
    return fig

def get_mesh_from_nifti(nifti_file):
    img = nib.load(nifti_file)
    data = img.get_fdata()
    verts, faces, _, _ = measure.marching_cubes(data, level=0.5)
    affine = img.affine
    verts = nib.affines.apply_affine(affine, verts)
    return verts, faces


def plot_DISTALmini(fig = None, coords = None, radius = 1): # this plot STN, GPe and GPi w/ coords if needed
    if fig is None:
        fig = go.Figure()
    fig = plot_3Dmesh(fig, region_input = 'STN')
    fig = plot_3Dmesh(fig, region_input = 'GPi')
    fig = plot_3Dmesh(fig, region_input = 'GPe')
    if coords is not None:
        fig = show_recordings(fig, coords, radius = radius)
    return fig


def plot_3Dmesh(fig = None, region_input = 'STN', atlas = 'DISTAL'):
    if isinstance(region_input, str):
        name_region = region_input
        if name_region == "STN":
            color = 'orange'
            nifti_file = os.path.join(BASE_DIR, f"resources/{atlas}/STN.nii.gz")
        elif name_region == "GPi":
            color = 'blue'
            nifti_file = os.path.join(BASE_DIR, f"resources/{atlas}/GPi.nii.gz")
        elif name_region == "GPe":
            color = 'green'
            nifti_file = os.path.join(BASE_DIR, f"resources/{atlas}/GPe.nii.gz")
        else:
            raise ValueError(f"Unknown region name: {name_region}")
    elif isinstance(region_input, dict):
        nifti_file = region_input.get('nifti_file')
        color = region_input.get('color', 'gray')  # Default color if not provided
        name_region = region_input.get('name', 'Unknown Region')
    else:
        raise TypeError("Input must be a string (region name) or a dictionary with 'nifti_file', 'color', and 'name'.")
    
       
        
    verts, faces = get_mesh_from_nifti(nifti_file)
    x, y, z = zip(*verts)
    i, j, k = zip(*faces)
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Mesh3d(
    x=x,
    y=y,
    z=z,
    i=i,
    j=j,
    k=k,
    color= color,
    opacity=0.2,
    name= name_region
    ))          
    
        
    # Set plot layout
    fig.update_layout(
        scene=dict(
            xaxis_title='MNI X [mm]',
            yaxis_title='MNI Y [mm]',
            zaxis_title='MNI Z [mm]',
            #aspectmode='cube'  # Ensure equal scaling on axes

        ),
    )
    return fig



def create_spheres(coords, radius=1):
    d = np.pi / 32  # Increment for theta and phi

    theta, phi = np.mgrid[0:np.pi + d:d, 0:2 * np.pi + d:d]
    # Convert to Cartesian coordinates
    x = radius * np.sin(theta) * np.cos(phi) + coords[0]
    y = radius * np.sin(theta) * np.sin(phi) + coords[1]
    z = radius * np.cos(theta) + coords[2]

    return x, y, z


# plot lfp continuous data
def plot_continuous_data(data_run, output_path = None, show = False):
    Run = data_run['run_details']['Run'].values[0]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data_run['data']['time'],
        y=data_run['data']['erna_cont'],
        mode="lines",
        name="LFP Data",
        line=dict(color='black'),
    ))

    # Add the second trace (line with secondary y-axis)
    fig.add_trace(go.Scatter(
        x=data_run['data']['timebursts'],
        y=data_run['data']['stim_amp'],
        mode="lines",
        name="Stim. amp [mA]",
        line=dict(color='red'),
        yaxis='y2'  # Specify that this trace uses the secondary y-axis
    ))

    # Update layout to add secondary y-axis and remove grid lines
    fig.update_layout(
        xaxis_title='time [GTC]',
        yaxis_title='LFP [µV]',
        title=f'ERNA Recording [run = {Run}]',
        xaxis=dict(
            showgrid=False  # Disable grid lines on the x-axis
        ),
        yaxis=dict(
            showgrid=False,  # Disable grid lines on the primary y-axis
            title='LFP [µV]'  # Title for the primary y-axis
        ),
        yaxis2=dict(
            overlaying='y',  # Overlay this y-axis on the primary y-axis
            side='right',    # Place this y-axis on the right side
            showgrid=False,   # Disable grid lines on the secondary y-axis
            title = "Stim. amp [mA]",
            titlefont=dict(color="red"),  # Set the title font color to red
            tickfont=dict(color="red"),   # Set the tick label font color to red            
        )
    )
    
    if output_path is not None:
        print("Printing figure...")
        #fig.write_html(output_path + ".html") 
        #print(f"Saved html figure in {output_path + '.html'}")
        pio.write_image(fig, output_path + ".svg")
        print(f"Saved svg figure in {output_path + '.svg'}")
        pio.write_image(fig, output_path + ".png")
        print(f"Saved svg figure in {output_path + '.png'}")       
    if show:
        fig.show()
    return 
# --------------------------------------------
# plot 3D location
def plot_stimChannelLocation(data_run, radius = 1,  output_path = None, show = False):
    stimChannel = data_run['stimChannel']
    Contact = stimChannel['name']
    coords = stimChannel['coords']['mni']
    x,y,z = coords[0]

    fig = plot_DISTALmini(coords = coords, radius = radius)
    
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.75, y=-2, z=1.75)
    )
        
    fig.update_layout(scene_camera = camera,
        title=f'Channel: {Contact} ({x:.2f},{y:.2f},{z:.2f}) [mm]',
                #aspectmode='cube'  # Ensure equal scaling on axes
        margin=dict(l=0, r=0, b=30, t=30),  # Remove padding
        scene=dict(aspectmode='data')  # Ensure equal scaling across axes
    )
    
    if output_path is not None:
        print("Printing figure...")
        #fig.write_html(output_path + ".html") 
        #print(f"Saved html figure in {output_path + '.html'}")
        pio.write_image(fig, output_path + ".svg")
        print(f"Saved svg figure in {output_path + '.svg'}")
        pio.write_image(fig, output_path + ".png")
        print(f"Saved svg figure in {output_path + '.png'}")       
    if show:
        fig.show()
    return 
# --------------------------------------------
# plot ERNA IBI matrix w/ average ERNA

def plot_IBImatrix(data_run, output_path = None, show = False):
    Run = data_run['run_details']['Run'].values[0]
    Task = data_run['run_details']['Task'].values[0]
    
    
    fig = make_subplots(rows=1, cols=2, 
            subplot_titles=('IBI Heatmap', 'Average IBI'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}]]
            )

    ibi_matrix, bursts, time_values, stim_vec = fetch_IBImatrix(data_run)

    # Create the heatmap
    fig.add_trace(go.Heatmap(
        z=ibi_matrix,
        x=bursts,
        y=time_values,
        colorscale='Spectral',  # Choose a colormap
        colorbar=dict(
            title='LFP [µV]',
            titlefont=dict(size=12),
            tickfont=dict(size=10)
        )
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=bursts,
        y=stim_vec,
        yaxis="y2",
        line=dict(color='red'),
        showlegend=False  # Remove legend
    ), row=1, col=1, secondary_y=True)

    # Set up the colormap 
    cm = plt.get_cmap('hot')


    if Task == "stimScan":
        target_stim_amp = [data_run['run_details']['Amp'].values[0]]
    elif Task == "ampRamp":
        target_stim_amp = np.arange(data_run['run_details']['Stim_step'].values[0],data_run['run_details']['Stim_target'].values[0] + 0.3,data_run['run_details']['Stim_step'].values[0])
        target_stim_amp = target_stim_amp[target_stim_amp >= 0]

    norm = plt.Normalize(vmin=1, vmax=len(target_stim_amp)) 
    for i,amp in enumerate(target_stim_amp):
        # Create a mask for the target stimulation amplitude
        mask = np.array(stim_vec) == amp
        filtered_ibi_matrix = ibi_matrix[:, mask]  # Filter ibi_matrix by the mask

        # Calculate average IBI over bursts for the target stimulation amplitude
        average_ibi = np.mean(filtered_ibi_matrix, axis=1)

        # Convert Matplotlib colormap to Plotly-compatible color
        rgba = cm(norm(i))  # RGBA color tuple
        color = f'rgba({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)}, {rgba[3]})'
        
        
        # Create the average IBI plot
        fig.add_trace(go.Scatter(
            x=time_values,
            y=average_ibi,
            mode="lines",
            line=dict(width=4, color=color),  # Assign a unique color
            name=f"{amp} mA",

        ), row=1, col=2)

    # Update axes
    fig.update_xaxes(title_text="Burst [id]", row=1, col=1)
    fig.update_yaxes(title_text="Time after burst [ms]" ,row=1, col=1)
    fig.update_xaxes(title_text="Time after burst [ms]",showgrid=True, dtick=5, row=1, col=2)
    fig.update_yaxes(title_text="LFP [µV]", showgrid=False, row=1, col=2)
    fig.update_layout(
        title=f'ERNA after burst [Run={Run}]',
        legend=dict(x=0.8,y=0.9),
        yaxis2=dict(
            title="Stim. Amp [mA]", 
            overlaying="y",  
            side="right",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            showgrid=False,
            showline=True, linecolor='black'# No gridlines for secondary y-axis in subplot 1
        ),
        
    )
    
    if output_path is not None:
        print("Printing figure...")
        #fig.write_html(output_path + ".html") 
        #print(f"Saved html figure in {output_path + '.html'}")
        pio.write_image(fig, output_path + ".svg")
        print(f"Saved svg figure in {output_path + '.svg'}")
        pio.write_image(fig, output_path + ".png")
        print(f"Saved svg figure in {output_path + '.png'}")   
    if show:
        fig.show()

    return



def animate_IBIavg(data_run, range_y = None, output_path = None, show = False):
    ibi_matrix, bursts, time_values, stim_vec = fetch_IBImatrix(data_run)

    # Create the DataFrame
    df = pd.DataFrame(ibi_matrix.T, index=bursts, columns=time_values)

    # Melt the DataFrame
    melted_df = df.reset_index().melt(id_vars='index', var_name='time_values', value_name='lfp')
    melted_df.rename(columns={'index': 'bursts'}, inplace=True)

    # Get min and max of lfp for consistent y-axis
    if range_y is None:
        lfp_min = melted_df['lfp'].min()
        lfp_max = melted_df['lfp'].max()
        range_y=[lfp_min, lfp_max]
    # Create the initial figure
    fig = px.line(melted_df, x="time_values", y="lfp", animation_frame="bursts" ,
                    range_y= range_y)

    # Increase the line width
    fig.update_traces(line=dict(width=4))  # Adjust the width here (4 is an example)

    # Adjust layout size and reduce title font size
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="LFP (µV)",
        title_font_size=14,  # Reduce title font size for readability
        width=600,  # Increase plot width
        height=400,  # Increase plot height
        transition={'duration': 200},
    )

    # Define an update to the title for each frame
    frames = fig.frames
    for i, frame in enumerate(frames):
        current_stim = stim_vec[i] if i < len(stim_vec) else "N/A"  # To avoid index errors
        frame['layout'] = {
            'title': f"fIBI Average Animation | Burst {bursts[i]} | Stim: {current_stim}"
        }

    # Ensure the animation displays correctly and titles are updated
    fig.update_layout(updatemenus=[{
        'buttons': [
            {
                'args': [None, {"frame": {"duration": 100, "redraw": True}, 
                                "fromcurrent": True, "transition": {"duration": 0}}],
                'label': 'Play',
                'method': 'animate'
            },
            {
                'args': [[None], {"frame": {"duration": 0, "redraw": True}, 
                                    "mode": "immediate", "transition": {"duration": 0}}],
                'label': 'Pause',
                'method': 'animate'
            }
        ],
        'direction': 'left',
        'pad': {'r': 10, 't': 87},
        'showactive': False,
        'type': 'buttons',
        'x': 0.1,
        'xanchor': 'right',
        'y': 0,
        'yanchor': 'top'
    }])
    
    
    if output_path is not None:
        print("Printing animation...")
        fig.write_html(output_path + ".html") 
        print(f"Saved html animation in {output_path + '.html'}")
    
    if show:
        fig.show()
    return 


# --------------------------------------------
import math




# plot ERNA IPI avg with color coding of pulse id

def plot_IPIavg(data_run, device = "argus",range_y = None, output_path=None, show=False):
    # Run details
    stim_freq = data_run['run_details']['Freq'].values[0]
    Task = data_run['run_details']['Task'].values[0]
    Burst_mode = data_run['run_details']['Burst_mode'].values[0]
    WIDEBAND = load_config(config_name=device).WIDEBAND
    
    if Burst_mode:
    
        # Determine target stimulation amplitudes
        if Task == "stimScan":
            target_stim_amp = [data_run['run_details']['Amp'].values[0]]
        elif Task == "ampRamp":
            target_stim_amp = np.arange(
                data_run['run_details']['Stim_step'].values[0],
                data_run['run_details']['Stim_target'].values[0] + 0.3,
                data_run['run_details']['Stim_step'].values[0]
            )
            target_stim_amp = target_stim_amp[target_stim_amp >= 0]
        
        # Fetch IPI matrix and related data
        ipi_matrix, time_matrix, bursts, npulses_inburst, stim_vec = fetch_IPImatrix(data_run)

        # Number of amplitudes
        n_amps = len(target_stim_amp)

        # Automatically calculate the number of rows and columns
        n_cols = math.ceil(math.sqrt(n_amps))  # Best number of columns based on sqrt
        n_rows = math.ceil(n_amps / n_cols)    # Best number of rows
        
        # Define the size of individual subplots (adjustable)
        subplot_width = 600  # Width of each subplot
        subplot_height = 600  # Decreased height of each subplot

        # Total figure size based on number of rows and columns
        total_width = subplot_width * n_cols
        total_height = subplot_height * n_rows

        # Create subplots layout in a grid format
        fig = make_subplots(
            rows=n_rows, cols=n_cols,  # Automatically calculated grid
            shared_xaxes=False,        # Optional: Share x-axes between rows
            shared_yaxes=False,        # Optional: Share y-axes between columns
            horizontal_spacing=0.06,    # Adjust spacing between columns
            vertical_spacing=0.06,      # Decreased spacing between rows
            subplot_titles=[f"Stim: {amp}" for amp in target_stim_amp]
        )

        # Set up the colormap (using 'hot' colormap)
        cm = plt.get_cmap('hot')
        norm = plt.Normalize(vmin=1, vmax=np.max(npulses_inburst))
        
        if range_y is None:
            # Calculate global y-axis range across all amplitudes
            global_min_ipi = np.inf
            global_max_ipi = -np.inf

            for amp in target_stim_amp:
                mask = np.array(stim_vec) == amp
                filtered_ipi_matrix = np.array(ipi_matrix)[mask, ...]  # Filter ipi_matrix by the mask

                # Calculate the mean IPI for this amplitude
                mean_ipi = np.mean(filtered_ipi_matrix, axis=0)  # Shape (n_pulses, n_ipi)

                # Update global min and max
                global_min_ipi = min(global_min_ipi, mean_ipi.min())
                global_max_ipi = max(global_max_ipi, mean_ipi.max())

            # Expand the global y-axis range slightly for better visualization
            global_min_ipi *= 1.1
            global_max_ipi *= 1.1
            range_y = [global_min_ipi, global_max_ipi]

        for i, amp in enumerate(target_stim_amp, start=1):
            # Calculate the row and column position for the subplot
            row = (i - 1) // n_cols + 1
            col = (i - 1) % n_cols + 1

            # Create a mask for the target stimulation amplitude
            mask = np.array(stim_vec) == amp
            filtered_ipi_matrix = np.array(ipi_matrix)[mask, ...]  # Filter ipi_matrix by the mask

            # Calculate the mean IPI and corresponding time for each pulse across bursts
            mean_ipi = np.mean(filtered_ipi_matrix, axis=0)  # Shape (n_pulses, n_ipi)
            mean_time = np.mean(time_matrix, axis=0)  # Shape (n_pulses, n_time)

            n_pulses = mean_ipi.shape[0]

            # Add traces for each pulse ID in the subplot
            for pulse_id in range(n_pulses):
                # Convert Matplotlib colormap to Plotly-compatible color
                rgba = cm(norm(pulse_id))  # RGBA color tuple
                color = f'rgba({int(rgba[0] * 255)}, {int(rgba[1] * 255)}, {int(rgba[2] * 255)}, {rgba[3]})'
                
                # Add trace to the corresponding subplot in the grid
                fig.add_trace(go.Scatter(
                    x=mean_time[pulse_id],
                    y=mean_ipi[pulse_id],
                    mode='lines',  # Change mode to only include lines
                    line=dict(color=color),
                    name=f'Pulse {pulse_id + 1}',
                    showlegend=False  # Hide legends for simplicity
                ), row=row, col=col)  # Add trace to the appropriate subplot in the grid
            
            # Set y-axis range for the current subplot to the global range
            fig.update_yaxes(range=range_y, row=row, col=col)

            # Add vertical lines and rectangle (if needed) to the subplot
            fig.add_shape(
                type="line",
                x0=0, x1=0, y0=range_y[0], y1=range_y[1],
                line=dict(color='blue', width=2, dash='dash'),
                row=row, col=col
            )
            fig.add_shape(
                type="line",
                x0=1/stim_freq*1000, x1=1/stim_freq*1000, y0=range_y[0], y1=range_y[1],
                line=dict(color='blue', width=2, dash='dash'),
                row=row, col=col
            )
            fig.add_shape(
                type='rect',
                x0=0,
                x1=WIDEBAND[str(stim_freq)] * 1000,
                y0=range_y[0],
                y1=range_y[1],
                fillcolor='rgba(255, 0, 0, 0.1)',  # Red rectangle with transparency
                line=dict(color='red', width=0.3),
                row=row, col=col
            )

            # Set axis titles for the current subplot
            fig.update_xaxes(title_text='Time (ms)', row=row, col=col, title_standoff=2)  # Added padding
            fig.update_yaxes(title_text='Average IPI (microV)', row=row, col=col, title_standoff=2)  # Added padding

        # Add a global color bar for pulse_id
        colorbar_trace = go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(width=0, color='rgba(0,0,0,0)'),
            showlegend=False,
            marker=dict(colorscale='hot', colorbar=dict(title='Pulse ID', titleside='right', len=0.75)),
            name='Colorbar'
        )
        fig.add_trace(colorbar_trace, row=n_rows, col=n_cols)  # Add to the last subplot for color bar

        # Update the layout to set figure size
        fig.update_layout(
            width=total_width,      # Total width of the figure
            height=total_height,    # Total height of the figure
            title="IPI ",
            showlegend=False        # Hide the overall legend
        )

        # Save and/or show the plot
        if output_path is not None:
            print("Printing figure...")
            #fig.write_html(output_path + ".html") 
            #print(f"Saved html figure in {output_path + '.html'}")
            pio.write_image(fig, output_path + ".svg")
            print(f"Saved svg figure in {output_path + '.svg'}")
            pio.write_image(fig, output_path + ".png")
            print(f"Saved svg figure in {output_path + '.png'}")   
        if show:
            fig.show()
    else:
        Warning("No IPI plots when Burst mode is not activated!")
    return





def animate_IPIavg(data_run, output_path=None, show = False, range_y=None):
    ipi_matrix, time_matrix, _, npulses_inburst, stim_vec = fetch_IPImatrix(data_run)
    Task = data_run['run_details']['Task'].values[0]
    
    Burst_mode = data_run['run_details']['Burst_mode'].values[0]
    
    # Use a sequential color scale (e.g., Viridis)
    color_scale = px.colors.sequential.Viridis    
    if Burst_mode:

        # Determine target stimulation amplitudes
        if Task == "stimScan":
            target_stim_amp = [data_run['run_details']['Amp'].values[0]]
            color_indices = [0]
            #color_map = {[target_stim_amp] : color_scale[int(0)]}
        elif Task == "ampRamp":
            target_stim_amp = np.arange(
                data_run['run_details']['Stim_step'].values[0],
                data_run['run_details']['Stim_target'].values[0] + 0.3,
                data_run['run_details']['Stim_step'].values[0]
            )
            target_stim_amp = target_stim_amp[target_stim_amp >= 0]
            normalized_stim_amps = (target_stim_amp - np.min(target_stim_amp)) / (np.max(target_stim_amp) - np.min(target_stim_amp))
            # Normalize stimulation amplitudes to get indices for color mapping
            color_indices = (normalized_stim_amps * (len(color_scale) - 1)).astype(int)


        
        color_map = {amp: color_scale[color_indices[i]] for i, amp in enumerate(target_stim_amp)}
        # Create a mask for the target stimulation amplitude
        #unique_stim_amps = np.unique(stim_vec)  # Get unique stimulation amplitudes
        traces = []  # List to hold traces for different stimulation amplitudes




        for amp in target_stim_amp:
            mask = np.array(stim_vec) == amp
            filtered_ipi_matrix = np.array(ipi_matrix)[mask, ...]  # Filter ipi_matrix by the mask

            # Calculate the mean IPI and corresponding time for each pulse across bursts
            mean_ipi = np.mean(filtered_ipi_matrix, axis=0)  # Shape (n_pulses, n_ipi)
            time_values = np.mean(np.mean(time_matrix, axis=0), axis=0)  # Shape (n_time,)
            pulses = np.array(range(1, int(np.mean(npulses_inburst)) + 1))  # Shape (n_pulses,)

            # Ensure the shapes are compatible before creating DataFrame 
            assert mean_ipi.shape == (len(pulses), len(time_values)), "Shape mismatch between mean_ipi.T, pulses, and time_values"

            # Create the DataFrame
            df = pd.DataFrame(mean_ipi, index=pulses, columns=time_values)

            # Melt the DataFrame
            melted_df = df.reset_index().melt(id_vars='index', var_name='time_values', value_name='lfp')
            melted_df.rename(columns={'index': 'pulses'}, inplace=True)

            # Add a column for stimulation amplitude
            melted_df['stim_amp'] = amp

            traces.append(melted_df)

        # Concatenate all traces into a single DataFrame
        final_df = pd.concat(traces)

        # Get min and max of lfp for consistent y-axis
        if range_y is None:
            lfp_min = final_df['lfp'].min()
            lfp_max = final_df['lfp'].max()
            range_y = [lfp_min, lfp_max]

        # Create the initial figure
        fig = px.line(final_df, x="time_values", y="lfp", animation_frame="pulses",
                    color='stim_amp',  # Color by stim_amp
                    color_discrete_map=color_map,  # Map colors from the defined color mapping
                    range_y=range_y)

        # Increase the line width
        fig.update_traces(line=dict(width=4))  # Adjust the width here (4 is an example)

        # Adjust layout size and reduce title font size
        fig.update_layout(
            xaxis_title="Time (s)",
            yaxis_title="LFP (µV)",
            title_font_size=14,  # Reduce title font size for readability
            width=600,  # Increase plot width
            height=400,  # Increase plot height
            transition={'duration': 200},
        )

        # Define an update to the title for each frame
        frames = fig.frames
        for i, frame in enumerate(frames):
            current_pulse = final_df['pulses'].unique()[i] if i < len(final_df['pulses'].unique()) else "N/A"  # To avoid index errors
            frame['layout'] = {
                'title': f"IPI Average Animation | Pulse {current_pulse}"
            }

        # Ensure the animation displays correctly and titles are updated
        fig.update_layout(updatemenus=[{
            'buttons': [
                {
                    'args': [None, {"frame": {"duration": 300, "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 0}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate", "transition": {"duration": 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }])
        
        if output_path is not None:
            print("Printing animation...")
            fig.write_html(output_path + ".html") 
            print(f"Saved html animation in {output_path + '.html'}")
        
        if show:
            fig.show()
    else:
        Warning("No IPI plots when Burst mode is not activated!")
    return




def plot_IBIfeatures_dynamics(IBI_annot, ref = "time", stim_freq = None , output_path=None, show = False):
    if ref == "time":
        x = IBI_annot['burst_id']
        xlabel = "Burst ID"
    elif ref == "charge" and stim_freq is not None:
        x = IBI_annot['stim_amp'].cumsum()*(1/stim_freq)*1e-3 # nC
        xlabel = "Charge [nC]"

        
        
    
    # Create subplots: 1 row, 3 columns
    fig = make_subplots(
        rows=1, cols=3,
        shared_xaxes=True,  # Share the x-axis
        specs=[[{"secondary_y": True}, {"secondary_y": True}, {"secondary_y": True}]],  # Enable secondary y-axes
        #subplot_titles=["Amplitude vs Stim Amp", "Frequency vs Stim Amp", "Latency vs Stim Amp"]
    )

    # Subplot 1: Amplitude and Stim Amp
    fig.add_trace(go.Scatter(
        x=x,
        y=IBI_annot['amplitude'],
        mode='markers',
        marker=dict(color='black'),
        showlegend=False,# Remove legend
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x,
        y=IBI_annot['stim_amp'],
        yaxis="y2",
        line=dict(color='red'),
        showlegend=False  # Remove legend
    ), row=1, col=1, secondary_y=True)

    # Subplot 2: Frequency and Stim Amp
    fig.add_trace(go.Scatter(
        x=x,
        y=IBI_annot['frequency'],
        mode='markers',
        marker=dict(color='blue'),
        showlegend=False  # Remove legend
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=x,
        y=IBI_annot['stim_amp'],
        yaxis="y2",
        line=dict(color='red'),
        showlegend=False  # Remove legend
    ), row=1, col=2, secondary_y=True)

    # Subplot 3: Latency and Stim Amp
    fig.add_trace(go.Scatter(
        x=x,
        y=IBI_annot['latency'],
        mode='markers',
        marker=dict(color='purple'),
        showlegend=False  # Remove legend
    ), row=1, col=3)

    fig.add_trace(go.Scatter(
        x=x,
        y=IBI_annot['stim_amp'],
        yaxis="y2",
        line=dict(color='red'),
        showlegend=False  # Remove legend
    ), row=1, col=3, secondary_y=True)

    # Update layout for the subplots
    fig.update_layout(
        # Set the range for x-axis in all subplots (starting from 50) and remove gridlines
        xaxis=dict(title=xlabel, range=[0, None], showgrid=False, showline=True, linecolor='black'),  # No gridlines for x-axis in subplot 1
        xaxis2=dict(title=xlabel, range=[0, None], showgrid=False, showline=True, linecolor='black'),  # No gridlines for x-axis in subplot 2
        xaxis3=dict(title=xlabel, range=[0, None], showgrid=False, showline=True, linecolor='black'),  # No gridlines for x-axis in subplot 3

        yaxis=dict(title="Amplitude", showgrid=False, showline=True, linecolor='black'),  # No gridlines for primary y-axis in subplot 1
        yaxis2=dict(
            title="Stim. Amp [mA]", 
            overlaying="y",  
            side="right",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            showgrid=False,
            showline=True, linecolor='black'# No gridlines for secondary y-axis in subplot 1
        ),
        
        yaxis3=dict(title="Frequency", showgrid=False, showline=True, linecolor='black'),  # No gridlines for primary y-axis in subplot 2
        yaxis4=dict(
            title="Stim. Amp [mA]",  
            overlaying="y3",  
            side="right",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            showgrid=False,
            showline=True, linecolor='black'# No gridlines for secondary y-axis in subplot 2
        ),
        
        yaxis5=dict(title="Latency",range = [3,6], showgrid=False, showline=True, linecolor='black'), # No gridlines for primary y-axis in subplot 3           
        yaxis6=dict(
            title="Stim. Amp [mA]",  
            overlaying="y5",  
            side="right",
            titlefont=dict(color="red"),
            tickfont=dict(color="red"),
            showgrid=False,
            showline=True, linecolor='black'# No gridlines for secondary y-axis in subplot 3
        ),

        title="IBI feature dynamics",
        width = 1200,
        height = 400
    )

    
    if output_path is not None:
        print("Printing figure...")
        #fig.write_html(output_path + ".html") 
        #print(f"Saved html figure in {output_path + '.html'}")
        pio.write_image(fig, output_path + ".svg")
        print(f"Saved svg figure in {output_path + '.svg'}")
        pio.write_image(fig, output_path + ".png")
        print(f"Saved svg figure in {output_path + '.png'}")       
    if show:
        fig.show()
        
    return



def plot_IPIfeatures_dynamics(IPI_annot, timescale = "pulse", ref = None, stim_freq = None, output_path=None, show = False):
    # Create a mask for the target stimulation amplitude and no more than 10 pulses
    # Use parentheses to clarify operator precedence
    #IPI_annot = IPI_annot.loc[(IPI_annot['stim_amp'] == IPI_annot['stim_amp'].max()) & 
    #                        (IPI_annot['pulse_id'] <= 9)]   
    

        
        
    IPI_annot = IPI_annot.loc[(IPI_annot['stim_amp'] > 0) & (IPI_annot['pulse_id'] <= 9)]  
    
        
    if timescale == "pulse":
        xlabel = "Pulse ID"
        
        # Create subplots: 2 rows, 2 columns
        fig = make_subplots(
            rows=2, cols=3,
        )



        # Generalized pulse IDs and sort them
        pulse_ids = np.sort(IPI_annot['pulse_id'].unique())
        
        # Compute medians for amplitude and latency
        amplitude_medians = [np.median(IPI_annot.loc[IPI_annot['pulse_id'] == pid, 'amplitude']) for pid in pulse_ids]
        latency_medians = [np.median(IPI_annot.loc[IPI_annot['pulse_id'] == pid, 'latency']) for pid in pulse_ids]
        
        # Normalize amplitudes and latencies within each burst (relative to pulse_id = 0)
        IPI_annot = IPI_annot.copy()  # Avoid "setting a value on a copy of a slice" warning
        IPI_annot['amplitude_normalized'] = np.nan  # Use np.nan for initialization
        IPI_annot['latency_normalized'] = np.nan
        
        unique_burst_ids = IPI_annot['burst_id'].unique()
        amplitude_pulse_first = []
        amplitude_pulse_last = []
        latency_pulse_first = []
        latency_pulse_last = []
        
        for burst in unique_burst_ids:
            burst_data = IPI_annot.loc[IPI_annot['burst_id'] == burst]
            ampl_first = burst_data.loc[burst_data['pulse_id'] == 0, 'amplitude'].values
            ampl_last = burst_data.loc[burst_data['pulse_id'] == 9, 'amplitude'].values
            lat_first = burst_data.loc[burst_data['pulse_id'] == 0, 'latency'].values
            lat_last = burst_data.loc[burst_data['pulse_id'] == 9, 'latency'].values
            
            # store them for later
            amplitude_pulse_first.extend(ampl_first)
            latency_pulse_first.extend(lat_first)
            amplitude_pulse_last.extend(ampl_last)
            latency_pulse_last.extend(lat_last)         

                 
            if len(ampl_first) != 0:
                IPI_annot.loc[IPI_annot['burst_id'] == burst, 'amplitude_normalized'] = 10*np.log10(
                    IPI_annot.loc[IPI_annot['burst_id'] == burst, 'amplitude'] / ampl_first
                )
            if len(lat_first) != 0:   
                IPI_annot.loc[IPI_annot['burst_id'] == burst, 'latency_normalized'] = IPI_annot.loc[IPI_annot['burst_id'] == burst, 'latency'] - lat_first

        # Compute normalized medians
        amplitude_norm_medians = [np.median(IPI_annot.loc[IPI_annot['pulse_id'] == pid, 'amplitude_normalized']) for pid in pulse_ids]
        latency_norm_medians = [np.median(IPI_annot.loc[IPI_annot['pulse_id'] == pid, 'latency_normalized']) for pid in pulse_ids]
        
        # comapre first and last in subqequent bursts
        
        amplitude_pulse_last = amplitude_pulse_last[1:]
        amplitude_pulse_first = amplitude_pulse_first[:-1]
        latency_pulse_last = latency_pulse_last[1:]
        latency_pulse_first = latency_pulse_first[:-1]        
        
        # Subplot 1: Amplitude vs Pulse ID
        fig.add_trace(go.Violin(
            x=IPI_annot['pulse_id'] + 1,  # Adding jitter to x-axis
            y=IPI_annot['amplitude'],
            points='all',  # Show all points
            box_visible=True,
            line_color='black',
            name='Amplitude',
            showlegend=False,
            bandwidth=1
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=pulse_ids + 1,
            y=amplitude_medians,
            mode='lines+markers',
            line=dict(color='red'),
            name='Median Amplitude',
            showlegend=False
        ), row=1, col=1)

        # Subplot 2: Latency vs Pulse ID
        fig.add_trace(go.Violin(
            x=IPI_annot['pulse_id'] + 1,
            y=IPI_annot['latency'],
            points='all',
            box_visible=True,
            line_color='purple',
            name='Latency',
            showlegend=False,
            bandwidth=1
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=pulse_ids + 1,
            y=latency_medians,
            mode='lines+markers',
            line=dict(color='red'),
            name='Median Latency',
            showlegend=False
        ), row=1, col=2)
        
        
        # Subplot 3: Amp pre vs plost
        fig.add_trace(go.Scatter(
            x=amplitude_pulse_last,
            y=amplitude_pulse_first,
            mode='markers',
            marker=dict(
                color=unique_burst_ids,  # Color based on the third variable
                colorscale='Viridis',  # Choose a color scale
                showscale=False          # Show color scale
            ),
            name='Amplitude before vs after',
            showlegend=False
        ), row=1, col=3)
        
        # Adding the bisector line (y = x)
        #max_value = max(max(amplitude_pulse_last), max(amplitude_pulse_first))
        fig.add_trace(go.Scatter(
            x=[100,300],
            y=[100,300],
            mode='lines',
            showlegend=False,
            line=dict(color='red', dash='dash')  # Dashed red line for the bisector
        ), row=1, col=3)        

        # Subplot 4: Normalized Amplitude (w.r.t pulse_id = 0)
        fig.add_trace(go.Violin(
            x=IPI_annot['pulse_id'] + 1 ,
            y=IPI_annot['amplitude_normalized'],
            points='all',
            box_visible=True,
            line_color='black',
            name='Normalized Amplitude',
            showlegend=False,
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=pulse_ids + 1,
            y=amplitude_norm_medians,
            mode='lines+markers',
            line=dict(color='red'),
            name='Median Normalized Amplitude',
            showlegend=False
        ), row=2, col=1)

        # Subplot 5: Normalized Latency (w.r.t pulse_id = 0)
        fig.add_trace(go.Violin(
            x=IPI_annot['pulse_id'] + 1,
            y=IPI_annot['latency_normalized'],
            points='all',
            box_visible=True,
            line_color='black',
            name='Normalized Latency',
            showlegend=False,
        ), row=2, col=2)

        fig.add_trace(go.Scatter(
            x=pulse_ids + 1,
            y=latency_norm_medians,
            mode='lines+markers',
            line=dict(color='red'),
            name='Median Normalized Latency',
            showlegend=False
        ), row=2, col=2)
        
        # Subplot 6: Latemncy pre vs plost
        fig.add_trace(go.Scatter(
            x=latency_pulse_last,
            y=latency_pulse_first,
            mode='markers',
            marker=dict(
                color=unique_burst_ids,  # Color based on the third variable
                colorscale='Viridis',  # Choose a color scale
                showscale=False          # Show color scale
            ),            
            name='Latency before vs after',
            showlegend=False
        ), row=2, col=3)
        
        # Adding the bisector line (y = x)
        #max_value = max(max(latency_pulse_last), max(latency_pulse_first))
        fig.add_trace(go.Scatter(
            x=[3.5, 5.5],
            y=[3.5, 5.5],
            mode='lines',
            showlegend=False,
            line=dict(color='red', dash='dash')  # Dashed red line for the bisector
        ), row=2, col=3)  
        
        # Update layout
        fig.update_layout(
            xaxis1=dict(title=xlabel, range=[0.5, 10.5], showgrid=False, showline=True, linecolor='black', tickvals =np.arange(1,11)),
            xaxis2=dict(title=xlabel, range=[0.5, 10.5], showgrid=False, showline=True, linecolor='black', tickvals =np.arange(1,11)),
            xaxis3=dict(title="Amplitude 10th pulse pre-burst [µV]", showgrid=False, showline=True, linecolor='black'),
            
            xaxis4=dict(title=xlabel, range=[0.5, 10.5], showgrid=False, showline=True, linecolor='black', tickvals =np.arange(1,11)),
            xaxis5=dict(title=xlabel, range=[0.5, 10.5], showgrid=False, showline=True, linecolor='black', tickvals =np.arange(1,11)),
            xaxis6=dict(title="Latency pre 10th pulse pre-burst [ms]", range=[2.5, 6.5], showgrid=False, showline=True, linecolor='black'),
            
            yaxis1=dict(title="Amplitude [µV]", showgrid=False, showline=True, linecolor='black'),
            yaxis4=dict(title="Amplitude Norm. [dB]", showgrid=False, showline=True, linecolor='black'),
            yaxis2=dict(title="Latency [ms]", range=[2.5, 6.5], showgrid=False, showline=True, linecolor='black'),
            yaxis3=dict(title="Amplitude post 1st pulse post-burst [µV]", showgrid=False, showline=True, linecolor='black'),
            
            yaxis5=dict(title="Latency diff. [ms]", showgrid=False, showline=True, linecolor='black'),
            yaxis6=dict(title="Latency post 1st pulse post-burst [ms]", range=[2.5, 6.5], showgrid=False, showline=True, linecolor='black'),
            
            title="IPI Feature Dynamics (Raw and Normalized)",
            width=1000,
            height=800,
        )
    elif timescale == "burst":
        # one plot for 10-1 pulse over bursts and plto centered at 10 vs -5 and + 5
        unique_burst_ids = IPI_annot['burst_id'].unique()
        amplitude_ratio = []
        latency_ratio = []
        stim_amp = []
        

        for burst in unique_burst_ids:
            burst_data = IPI_annot.loc[IPI_annot['burst_id'] == burst]
            stim_amp.extend(burst_data['stim_amp'].unique())
            
            amplitude_pulse_first = burst_data.loc[burst_data['pulse_id'] == 0, 'amplitude'].values
            latency_pulse_first = burst_data.loc[burst_data['pulse_id'] == 0, 'latency'].values
            amplitude_pulse_last = burst_data.loc[burst_data['pulse_id'] == 9, 'amplitude'].values
            latency_pulse_last = burst_data.loc[burst_data['pulse_id'] == 9, 'latency'].values
            
            
            # append
            amplitude_ratio.extend(10*np.log10(amplitude_pulse_last/amplitude_pulse_first))   
            #latency_ratio.extend(10*np.log10(latency_pulse_last/latency_pulse_first))   
            #amplitude_ratio.extend(amplitude_pulse_last -amplitude_pulse_first)   
            latency_ratio.extend(latency_pulse_last-latency_pulse_first)               
            
        if ref == "time":
            x = unique_burst_ids
            xlabel = "Burst ID"
        elif ref == "charge" and stim_freq is not None:
            x = np.cumsum(stim_amp)*(1/stim_freq)*1e-3 # nC
            xlabel = "Charge [nC]"
        
        
        # Create subplots: 1 row, 3 columns
        fig = make_subplots(
            rows=1, cols=2,
            shared_xaxes=True,  # Share the x-axis
            specs=[[{"secondary_y": True}, {"secondary_y": True}]],  # Enable secondary y-axes
            #subplot_titles=["Amplitude vs Stim Amp", "Frequency vs Stim Amp", "Latency vs Stim Amp"]
        )

        # Subplot 1: Amplitude and Stim Amp
        fig.add_trace(go.Scatter(
            x=x,
            y=amplitude_ratio,
            mode='markers',
            marker=dict(color='black'),
            showlegend=False,# Remove legend
        ), row=1, col=1)
        
        
        fig.add_trace(go.Scatter(
            x=x,
            y=stim_amp,
            yaxis="y2",
            line=dict(color='red'),
            showlegend=False  # Remove legend
        ), row=1, col=1, secondary_y=True)

        fig.add_trace(go.Scatter(
            x=x,
            y=latency_ratio,
            mode='markers',
            marker=dict(color='black'),
            showlegend=False,# Remove legend
        ), row=1, col=2)
            
        fig.add_trace(go.Scatter(
            x=x,
            y=stim_amp,
            yaxis="y2",
            line=dict(color='red'),
            showlegend=False  # Remove legend
        ), row=1, col=2, secondary_y=True)
        
    # Update layout for the subplots
        fig.update_layout(
            # Set the range for x-axis in all subplots (starting from 50) and remove gridlines
            xaxis=dict(title=xlabel, range=[0, None], showgrid=False, showline=True, linecolor='black'),  # No gridlines for x-axis in subplot 1
            xaxis2=dict(title=xlabel, range=[0, None], showgrid=False, showline=True, linecolor='black'),  # No gridlines for x-axis in subplot 2
            xaxis3=dict(title=xlabel, range=[0, None], showgrid=False, showline=True, linecolor='black'),  # No gridlines for x-axis in subplot 3

            yaxis=dict(title="Amplitude 10th - 1st pulse [dB]", showgrid=False, showline=True, linecolor='black'),  # No gridlines for primary y-axis in subplot 1
            yaxis2=dict(
                title="Stim. Amp [mA]", 
                overlaying="y",  
                side="right",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
                showgrid=False,
                showline=True, linecolor='black'# No gridlines for secondary y-axis in subplot 1
            ),
            
            yaxis3=dict(title="Latency 10th - 1st pulse [ms]", showgrid=False, showline=True, linecolor='black'),  # No gridlines for primary y-axis in subplot 2
            yaxis4=dict(
                title="Stim. Amp [mA]",  
                overlaying="y3",  
                side="right",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
                showgrid=False,
                showline=True, linecolor='black'# No gridlines for secondary y-axis in subplot 2
            ),
            

            title="IPI feature dynamics"
        )
    # Show the figure
    if output_path is not None:
        print("Printing figure...")
        #fig.write_html(output_path + ".html") 
        #print(f"Saved html figure in {output_path + '.html'}")
        pio.write_image(fig, output_path + ".svg")
        print(f"Saved svg figure in {output_path + '.svg'}")
        pio.write_image(fig, output_path + ".png")
        print(f"Saved svg figure in {output_path + '.png'}")      
    if show:
        fig.show()
    return

